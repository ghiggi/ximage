#!/usr/bin/env python3
"""
Created on Tue Jul 11 11:42:30 2023

@author: ghiggi
"""
import numpy as np


def are_all_integers(arr, negative_allowed=True, zero_allowed=True):
    """
    Check if all values in the input numpy array are integers.

    Parameters
    ----------
    arr : (list, tuple, np.ndarray)
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
    else:
        if zero_allowed:
            return bool(np.all(np.logical_and(arr >= 0, is_integer)))
        else:
            return bool(np.all(np.logical_and(arr > 0, is_integer)))


def are_all_natural_numbers(arr, zero_allowed=False):
    """
    Check if all values in the input numpy array are natural numbers (>1).

    Parameters
    ----------
    arr : (list, tuple, np.ndarray)
       List, tuple or array of values to be checked.

    Returns
    -------
    bool
        True if all values in the array are natural numbers. False otherwise.

    """
    return are_all_integers(arr, negative_allowed=False, zero_allowed=zero_allowed)
