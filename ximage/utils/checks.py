# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024-2026 ximage developers
#
# This file is part of ximage.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""Utility functions for checking input values."""
import numpy as np


def are_all_integers(arr, negative_allowed=True, zero_allowed=True):
    """
    Check if all values in the input numpy array are integers.

    Parameters
    ----------
    arr : (list, tuple, numpy.ndarray)
       List, tuple or array of values to be checked.
    negative_allowed: bool, optional
        If False, return True only for integers >=1 (natural numbers)
    zero_allowed : bool, optional
        Used only if negative_allowed=False.
        Whether to consider 0 a valid value. The default is True.

    Returns
    -------
    bool
        True if all values in the array are integers, False otherwise.

    """
    # Convert to numpy array
    arr = np.asanyarray(arr)

    is_integer = np.isclose(arr, np.round(arr), atol=1e-12, rtol=1e-12)
    if negative_allowed:
        return bool(np.all(is_integer))
    if zero_allowed:
        return bool(np.all(np.logical_and(arr >= 0, is_integer)))
    return bool(np.all(np.logical_and(arr > 0, is_integer)))


def are_all_natural_numbers(arr, zero_allowed=False):
    """
    Check if all values in the input numpy array are natural numbers (>1).

    Parameters
    ----------
    arr : (list, tuple, numpy.ndarray)
       List, tuple or array of values to be checked.

    Returns
    -------
    bool
        True if all values in the array are natural numbers. False otherwise.

    """
    return are_all_integers(arr, negative_allowed=False, zero_allowed=zero_allowed)
