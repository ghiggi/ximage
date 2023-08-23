import numpy as np
import pytest


class SaneEqualityArray(np.ndarray):
    """Wrapper class for numpy array allowing deep equality tests on objects containing numpy arrays.

    From https://stackoverflow.com/a/14276901
    """

    def __new__(cls, array):
        """Create a new SaneEqualityArray from array only (instead of shape + type + array)."""
        return np.asarray(array).view(cls)

    def __eq__(self, other):
        return (
            isinstance(other, np.ndarray) and self.shape == other.shape and np.allclose(self, other)
        )


def pytest_configure():
    pytest.SaneEqualityArray = SaneEqualityArray
