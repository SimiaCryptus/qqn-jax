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

`pytest` is the only test tool worth using. Write a file starting with `test_`, write functions starting with `test_`, use plain `assert`.

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

That's it. Run those two before every commit and your code stays clean.

---

## 5. Publishing to PyPI

This is the scary part but it's only 3 commands. There are **two** PyPIs:
- **TestPyPI** — practice playground (`test.pypi.org`)
- **Real PyPI** — the real deal (`pypi.org`)

**Always test on TestPyPI first.**

### One-time setup
1. Make an account on https://pypi.org and https://test.pypi.org
2. Create an **API token** (Account Settings → API tokens). It looks like `pypi-AgEN...`
3. Don't paste tokens into files. `twine` will prompt you, or use env vars.

### Build the package

```bash
# Cleans old builds, then builds wheel + source dist into dist/
rm -rf dist/
python -m build
```

This creates two files in `dist/`:
- `qqn_jax-0.1.0-py3-none-any.whl` (the wheel — what people install)
- `qqn_jax-0.1.0.tar.gz` (source archive)

### Upload to TestPyPI first (practice)

```bash
twine upload --repository testpypi dist/*
# Username: __token__
# Password: <paste your TestPyPI token>
```

Then verify it installs:

```bash
pip install --index-url https://test.pypi.org/simple/ qqn-jax
```

### Upload to real PyPI (the real deal)

```bash
twine upload dist/*
# Username: __token__
# Password: <paste your real PyPI token>
```

Done. Now `pip install qqn-jax` works for everyone.

**Critical rule:** You can **never** re-upload the same version number. Bump `version = "0.1.0"` → `"0.1.1"` in `pyproject.toml` every single release.

