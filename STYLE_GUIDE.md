# Style Guide

This project uses a simple Python style focused on readable code, clear
docstrings, and consistent formatting.

## Formatting

Python code is formatted with Black.

The maximum line length is 79 characters.

Before committing, run:

```bash
pre-commit run --all-files
```

## Naming

Use standard Python naming conventions:

* Functions and variables: `snake_case`
* Classes: `PascalCase`
* Constants: `UPPER_CASE`
* Private helpers: leading underscore, for example `_estimate_background`

Use descriptive names when they make the code easier to read.

Good:

```python
total_counts = counts.sum()
background_scale = peak_volume / shell_volume
```

Avoid unclear names:

```python
x = counts.sum()
s = peak_volume / shell_volume
```

Short mathematical names such as `i`, `j`, `k`, `x`, `y`, `z`, `q`, `r`, or
`w` are fine when their meaning is clear.

## Imports

Group imports in this order:

1. Standard library
2. Third-party packages
3. Local imports

Example:

```python
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from package.module import helper_function
```

Avoid wildcard imports:

```python
from module import *
```

## Docstrings

Use NumPy-style docstrings for public modules, classes, functions, and methods.

Parameter, return value, yielded value, and attribute entries must include both
a variable name and a description.

Use this form:

```python
counts : int
    Total raw counts.
```

Do not write entries without descriptions:

```python
counts : int
```

Do not write descriptions without the variable name and type:

```python
Total raw counts.
```

Keep descriptions simple and direct.

Good:

```python
counts : int
    Total raw counts.
```

Avoid overly broad or indirect descriptions:

```python
counts : int
    Counts used by other parts of the processing workflow.
```

## Function Docstring Example

```python
def subtract_background(counts, background, scale):
    """
    Subtract a scaled background from measured counts.

    Parameters
    ----------
    counts : ndarray
        Measured raw counts in the peak region.
    background : ndarray
        Measured background counts.
    scale : float
        Scale factor applied to the background.

    Returns
    -------
    intensity : float
        Background-subtracted integrated intensity.
    """
    return counts.sum() - scale * background.sum()
```

## Return Values

Only include a `Returns` section when something is returned.

If a function does not return anything, omit the `Returns` section.

Do not write a `Returns` section only to say that the function returns `None`.

Good:

```python
def save_results(filename, results):
    """
    Save results to a file.

    Parameters
    ----------
    filename : str
        Output filename.
    results : dict
        Results to save.
    """
```

Avoid:

```python
def save_results(filename, results):
    """
    Save results to a file.

    Parameters
    ----------
    filename : str
        Output filename.
    results : dict
        Results to save.

    Returns
    -------
    None
        This function does not return anything.
    """
```

## Array Descriptions

When using arrays, include shape, units, or coordinate frame when they matter.

Good:

```python
q_lab : ndarray of shape (n_peaks, 3)
    Peak positions in the laboratory frame, in inverse angstroms.
```

Acceptable when no extra detail is needed:

```python
values : ndarray
    Input values.
```

## Comments

Keep comments simple and direct.

Use comments to explain useful context, not to repeat the code.

Good:

```python
# Scale the background by the relative integration volume.
scale = peak_volume / shell_volume
```

Avoid:

```python
# Set scale equal to peak_volume divided by shell_volume.
scale = peak_volume / shell_volume
```

Avoid comments that refer vaguely to other code.

Good:

```python
# Use a small value to avoid division by zero.
sigma = np.maximum(sigma, 1e-12)
```

Avoid:

```python
# Needed for downstream routines.
sigma = np.maximum(sigma, 1e-12)
```

## Numerical Code

For numerical routines, prefer clear code over compact code.

Document important details when they matter:

* Array shape
* Units
* Coordinate frame
* Normalization
* Masking behavior
* Invalid values

Use intermediate variables when they make the calculation easier to read.

Good:

```python
signal = counts - background_scale * background
variance = counts + background_scale**2 * background
sigma = np.sqrt(variance)
```

## Errors

Raise clear errors for invalid inputs.

Good:

```python
if counts.shape != background.shape:
    raise ValueError("counts and background must have the same shape")
```

## Tests

Add tests when changing behavior.

Numerical tests should include simple cases where the expected result is known.
