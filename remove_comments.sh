#!/usr/bin/env bash
# Remove all comments from all python files in this project
#
# This script strips comments from Python source files using Python's
# tokenizer, which correctly handles comments vs. strings that contain
# '#' characters. Docstrings are preserved by default.
#
# Usage:
#   ./remove_comments.sh [--dry-run] [--docstrings] [path ...]
#
# Options:
#   --dry-run      Show which files would be changed without modifying them.
#   --docstrings   Also remove module/class/function docstrings.
#   -h, --help     Show this help message.
#
# If no paths are given, the current directory is searched recursively.

set -euo pipefail

DRY_RUN=0
REMOVE_DOCSTRINGS=0
PATHS=()

print_help() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --docstrings)
            REMOVE_DOCSTRINGS=1
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do PATHS+=("$1"); shift; done
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            PATHS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#PATHS[@]} -eq 0 ]]; then
    PATHS=(".")
fi

# Locate a python interpreter.
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "Error: could not find a python interpreter." >&2
    exit 1
fi

# Collect Python files, skipping common virtualenv / build / VCS dirs.
mapfile -d '' FILES < <(
    find "${PATHS[@]}" \
        -type d \( \
            -name .git -o \
            -name .venv -o \
            -name venv -o \
            -name __pycache__ -o \
            -name build -o \
            -name dist -o \
            -name .tox -o \
            -name .mypy_cache -o \
            -name .pytest_cache \
        \) -prune -o \
        -type f -name '*.py' -print0
)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No Python files found."
    exit 0
fi

export REMOVE_DOCSTRINGS

changed=0
for file in "${FILES[@]}"; do
    # Produce a comment-stripped version on stdout.
    if ! stripped="$("$PYTHON" - "$file" <<'PYEOF'
import io
import os
import sys
import tokenize

path = sys.argv[1]
remove_docstrings = os.environ.get("REMOVE_DOCSTRINGS") == "1"

with open(path, "rb") as f:
    source_bytes = f.read()

try:
    tokens = list(tokenize.tokenize(io.BytesIO(source_bytes).readline))
except tokenize.TokenError as exc:
    sys.stderr.write(f"Skipping {path}: tokenize error: {exc}\n")
    sys.exit(2)

out_tokens = []
prev_toktype = tokenize.INDENT
for tok in tokens:
    toktype = tok.type
    tokstring = tok.string

    if toktype == tokenize.COMMENT:
        # Drop comments entirely.
        continue

    if (
        remove_docstrings
        and toktype == tokenize.STRING
        and prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.ENCODING)
    ):
        # Replace a docstring with an empty string literal to keep syntax valid.
        tok = tok._replace(string='""')

    out_tokens.append(tok)
    if toktype not in (tokenize.NL, tokenize.COMMENT):
        prev_toktype = toktype

try:
    result = tokenize.untokenize(out_tokens)
except Exception as exc:
    sys.stderr.write(f"Skipping {path}: untokenize error: {exc}\n")
    sys.exit(2)

if isinstance(result, bytes):
    sys.stdout.buffer.write(result)
else:
    sys.stdout.buffer.write(result.encode("utf-8"))
PYEOF
    )"; then
        echo "Warning: failed to process $file (left unchanged)." >&2
        continue
    fi

    # Compare with the original to see if anything changed.
    if [[ "$stripped" == "$(cat "$file")" ]]; then
        continue
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "Would strip comments: $file"
    else
        printf '%s' "$stripped" > "$file"
        echo "Stripped comments: $file"
    fi
    changed=$((changed + 1))
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete. $changed file(s) would be modified."
else
    echo "Done. $changed file(s) modified."
fi