# Contributing

Contributions should keep the code readable, tested, and consistent.

## Setup

Clone the repository and create a Python environment.

```bash
git clone <repository-url>
cd <repository-name>
python -m venv .venv
```

Activate the environment.

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package and development dependencies using the project setup
instructions.

## Pre-commit

This project uses pre-commit for formatting and linting.

Install pre-commit:

```bash
pip install pre-commit
```

Install the hooks:

```bash
pre-commit install
```

Run all checks before committing:

```bash
pre-commit run --all-files
```

All submitted code should pass the pre-commit checks.

## Code Style

Use Black for formatting.

The maximum line length is 79 characters.

Use NumPy-style docstrings for public functions, classes, methods, and modules.

Docstring entries must include a variable name, type, and description.

Good:

```python
counts : int
    Total raw counts.
```

Avoid:

```python
counts : int
```

Keep comments and descriptions simple and direct.

Good:

```python
scale : float
    Scale factor applied to the background.
```

Avoid:

```python
scale : float
    Scale factor used by the broader background correction machinery.
```

## Tests

Add or update tests when changing behavior.

Run the tests before submitting changes:

```bash
pytest
```

For numerical code, include simple cases where the expected result is known.

## Pull Requests

Before opening a pull request:

1. Run pre-commit.
2. Run the relevant tests.
3. Update docstrings if behavior changed.
4. Keep the change focused when possible.

A good pull request description explains:

* What changed
* Why it changed
* How it was tested

## Commit Messages

Use clear commit messages.

Good:

```text
Add background scale validation
```

Avoid:

```text
Fix stuff
```

## Numerical Changes

For numerical algorithms, keep assumptions explicit.

Document details when they matter:

* Units
* Array shape
* Coordinate frame
* Normalization
* Masking behavior
* Invalid values

Prefer simple, direct explanations.
