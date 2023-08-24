import dask.array
import numpy as np
import pytest
import xarray as xr


class SaneEqualityArray(np.ndarray):
    """Wrapper class for numpy array allowing deep equality tests on objects containing numpy arrays.

    From https://stackoverflow.com/a/14276901
    """

    def __new__(cls, array):
        """Create a new SaneEqualityArray from array only (instead of shape + type + array)."""
        return np.asarray(array).view(cls)

    def __eq__(self, other):
        return (
            isinstance(other, np.ndarray)
            and self.shape == other.shape
            and np.array_equal(self, other, equal_nan=True)
        )


def apply_to_all_array_types(func, array, *args, **kwargs):
    """Apply a function to np.ndarray, dask.Array, and xr.DataArray."""

    np_array = np.array(array)
    dask_array = dask.array.from_array(array)
    xr_array = xr.DataArray(array)

    for x_array in [np_array, dask_array, xr_array]:
        func(x_array, *args, **kwargs)


def pytest_configure():
    pytest.SaneEqualityArray = SaneEqualityArray
    pytest.apply_to_all_array_types = apply_to_all_array_types
