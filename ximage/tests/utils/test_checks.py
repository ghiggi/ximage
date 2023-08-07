import numpy as np

from ximage.utils import checks


def test_are_all_integers():
    """Test are_all_integers"""

    # Test cases with negative_allowed=True and zero_allowed=True
    assert checks.are_all_integers([1, 2, 3])
    assert checks.are_all_integers([0, -1, -2, -3])
    assert not checks.are_all_integers([1.5, 2.5, 3.5])
    assert not checks.are_all_integers([1, 2, 3.5])

    # Test cases with negative_allowed=False and zero_allowed=True
    assert checks.are_all_integers([1, 2, 3], negative_allowed=False)
    assert not checks.are_all_integers([0, -1, -2, -3], negative_allowed=False)
    assert not checks.are_all_integers([1.5, 2.5, 3.5], negative_allowed=False)
    assert not checks.are_all_integers([1, 2, 3.5], negative_allowed=False)

    # Test cases with negative_allowed=False and zero_allowed=False
    assert checks.are_all_integers([1, 2, 3], negative_allowed=False, zero_allowed=False)
    assert not checks.are_all_integers([0, -1, -2, -3], negative_allowed=False, zero_allowed=False)
    assert not checks.are_all_integers([1.5, 2.5, 3.5], negative_allowed=False, zero_allowed=False)
    assert not checks.are_all_integers([1, 2, 3.5], negative_allowed=False, zero_allowed=False)

    # Test cases with numpy arrays
    assert checks.are_all_integers(np.array([1, 2, 3]))
    assert not checks.are_all_integers(np.array([1.5, 2.5, 3.5]))
    assert checks.are_all_integers(np.array([0, -1, -2, -3]))
    assert not checks.are_all_integers(np.array([1, 2, 3.5]))

    # Test cases with tuples
    assert checks.are_all_integers((1, 2, 3))
    assert not checks.are_all_integers((1.5, 2.5, 3.5))
    assert checks.are_all_integers((0, -1, -2, -3))
    assert not checks.are_all_integers((1, 2, 3.5))


def test_are_all_natural_numbers():
    """Test are_all_natural_numbers"""

    # Test cases with zero_allowed=False
    assert checks.are_all_natural_numbers([1, 2, 3])
    assert not checks.are_all_natural_numbers([0, 2, 3])
    assert not checks.are_all_natural_numbers([2.5, 3.5, 4.5])

    # Test cases with zero_allowed=True
    assert checks.are_all_natural_numbers([1, 2, 3], zero_allowed=True)
    assert checks.are_all_natural_numbers([0, 2, 3], zero_allowed=True)
    assert not checks.are_all_natural_numbers([2.5, 3.5, 4.5], zero_allowed=True)

    # Test cases with numpy arrays
    assert checks.are_all_natural_numbers(np.array([1, 2, 3]))
    assert not checks.are_all_natural_numbers(np.array([0, 2, 3]))
    assert not checks.are_all_natural_numbers(np.array([2.5, 3.5, 4.5]))

    # Test cases with tuples
    assert checks.are_all_natural_numbers((1, 2, 3))
    assert not checks.are_all_natural_numbers((0, 2, 3))
    assert not checks.are_all_natural_numbers((2.5, 3.5, 4.5))
