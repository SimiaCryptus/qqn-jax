## 1. Virtual Environments (venv)

A venv is just an isolated folder of packages so you don't pollute your system Python. **Always use one.**

```bash
# Create it (do this ONCE per project)
python3 -m venv .venv

# Activate it (do this EVERY new terminal)
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (cmd)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

# When you're done / want out
deactivate
```

You'll know it's active because your prompt shows `(.venv)`.

**Pro tip:** Add `.venv/` to `.gitignore` so you never commit it.

---

## 2. Installing Your Project

`pip install -e ".[dev]"` - Here's what that does:

- `-e` = "editable" — your code changes take effect without reinstalling. **Always use this in dev.**
- `.` = install the package in the current directory.
- `[dev]` = also install the optional "dev" dependencies (test tools, etc).

```bash
# With your venv active:
pip install -e ".[dev]"
```

---

## 3. Testing (pytest)

`pytest` is the only test tool worth using. Write a file starting with `test_`, write functions starting with `test_`,
use plain `assert`.

Run it:

```bash
# Run all tests
pytest

# Verbose (see each test name)
pytest -v

# Run one file
pytest tests/test_solver.py

# Run one specific test
pytest tests/test_solver.py::test_jit_compatible

# Stop at first failure (saves time)
pytest -x

# With coverage report (how much code your tests hit)
pip install pytest-cov
pytest --cov=qqn_jax
```

---

## 4. Linting / Formatting (ruff — do this, it's painless)

`ruff` replaces `black`, `flake8`, `isort`, and more. One tool, instant.

```bash
# Auto-format all your code
ruff format .

# Find problems (and auto-fix what it can)
ruff check . --fix
```

---

## 5. Type Checking (mypy / pyright)

Static type checking catches bugs before you run your code. Add type hints to
your functions, then let a checker verify them.

### 5.1 Declaring Your Package as Typed (PEP 561)

If you ship type hints, you **must** tell tools they exist by adding a marker
file. Without it, downstream users won't get type checking against your package.

```bash
# Create an empty marker file inside your package directory
touch qqn_jax/py.typed
```

Then make sure it's included in your build. In `pyproject.toml`:

```toml
# setuptools
[tool.setuptools.package-data]
qqn_jax = ["py.typed"]
# OR for hatchling
[tool.hatch.build.targets.wheel]
packages = ["qqn_jax"]
```

**Pro tip:** The `py.typed` file is empty — its mere presence is the signal.

### 5.2 Running a Type Checker

```bash
# mypy (the classic)
pip install mypy
mypy qqn_jax
# pyright (fast, used by VS Code / Pylance)
pip install pyright
pyright
```

### 5.3 Configuring mypy

Put config in `pyproject.toml` so everyone uses the same settings:

```toml
[tool.mypy]
python_version = "3.10"
strict = true                # turn on all the good checks
warn_unused_ignores = true   # flag stale "# type: ignore" comments
warn_redundant_casts = true
files = ["qqn_jax", "tests"]
# Silence third-party libs that lack stubs
[[tool.mypy.overrides]]
module = ["some_untyped_lib.*"]
ignore_missing_imports = true
```

### 5.4 Configuring pyright

```toml
[tool.pyright]
include = ["qqn_jax", "tests"]
typeCheckingMode = "strict"
pythonVersion = "3.10"
```

### 5.5 Inline Escape Hatches

Sometimes the checker is wrong (or a library is untyped). Suppress narrowly:

```python
result = legacy_call()  # type: ignore[no-untyped-call]
```

**Always** pin the specific error code in brackets so you don't accidentally
hide *other* real errors on that line.

### 5.6 Add Type Checking to Dev Dependencies

Wire it into your optional `[dev]` extras so `pip install -e ".[dev]"` grabs it:

```toml
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
]
```