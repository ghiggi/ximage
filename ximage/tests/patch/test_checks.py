import pytest

from ximage.patch import checks


# Utils functions ##############################################################


def patch_and_kernel_size_routine(tested_function):
    """Routine used by test_check_patch_size and test_check_kernel_size."""

    dimension_names = ["x", "y"]
    shape = (10, 10)
    patch_size_expected = {"x": 2, "y": 2}

    # Test int, list, tuple, and dict patch_size
    for patch_size in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
        patch_size_returned = tested_function(patch_size, dimension_names, shape)
        assert patch_size_returned == patch_size_expected

    # Test full dimension size
    patch_size = -1
    patch_size_expected = {"x": 10, "y": 10}
    patch_size_returned = tested_function(patch_size, dimension_names, shape)
    assert patch_size_returned == patch_size_expected

    # Check ValueError: invalid dimension number, invalid dimension name,
    # missing dimension name, and invalid patch_size value (non-integer or too large)
    for patch_size in [[2, 2, 2], {"x": 2, "z": 2}, {"x": 2}, 0, 1.5, 20]:
        with pytest.raises(ValueError):
            tested_function(patch_size, dimension_names, shape)

    # Check invalid patch_size type
    patch_size = "2"
    with pytest.raises(TypeError):
        tested_function(patch_size, dimension_names, shape)


def buffer_and_padding_routine(tested_function):
    """Routine used by test_check_buffer and test_check_padding."""

    dimension_names = ["x", "y"]
    shape = (10, 10)
    buffer_expected = {"x": 2, "y": 2}

    # Test int, list, tuple, and dict buffer
    for buffer in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
        buffer_returned = tested_function(buffer, dimension_names, shape)
        assert buffer_returned == buffer_expected

    # Check invalid values (non-integer)
    buffer = 1.5
    with pytest.raises(ValueError):
        tested_function(buffer, dimension_names, shape)


# Tests for public functions ###################################################


def test_check_patch_size():
    """Test check_patch_size."""

    patch_and_kernel_size_routine(checks.check_patch_size)


def test_check_kernel_size():
    """Test check_kernel_size."""

    patch_and_kernel_size_routine(checks.check_kernel_size)


def test_check_buffer():
    """Test check_buffer."""

    buffer_and_padding_routine(checks.check_buffer)


def test_check_padding():
    """Test check_padding."""

    buffer_and_padding_routine(checks.check_padding)


def test_check_partitioning_method():
    """Test check_partitioning_method"""

    # Check valid values
    for method in [None, "sliding", "tiling"]:
        assert checks.check_partitioning_method(method) == method

    # Check invalid values
    with pytest.raises(ValueError):
        checks.check_partitioning_method("invalid")

    # Check invalid type
    with pytest.raises(TypeError):
        checks.check_partitioning_method(1)


def test_check_stride():
    """Test check_stride."""

    dimension_names = ["x", "y"]
    shape = (10, 10)

    # Check no partitioning method
    partitioning_method = None
    assert checks.check_stride(None, dimension_names, shape, partitioning_method) == None

    # Check default stride
    partitioning_method = "tiling"
    stride_expected = {"x": 0, "y": 0}
    assert checks.check_stride(None, dimension_names, shape, partitioning_method) == stride_expected

    partitioning_method = "sliding"
    stride_expected = {"x": 1, "y": 1}
    assert checks.check_stride(None, dimension_names, shape, partitioning_method) == stride_expected

    # Check with given stride
    stride_expected = {"x": 2, "y": 2}
    for stride in [2, [2, 2], (2, 2), {"x": 2, "y": 2}]:
        for partitioning_method in ["sliding", "tiling"]:
            stride_returned = checks.check_stride(
                stride, dimension_names, shape, partitioning_method
            )
            assert stride_returned == stride_expected

    # Check invalid values
    # Tiling only integers
    checks.check_stride(-1, dimension_names, shape, "tiling")
    checks.check_stride(0, dimension_names, shape, "tiling")
    with pytest.raises(ValueError):
        checks.check_stride(1.5, dimension_names, shape, "tiling")

    # Sliding only positive integers
    for stride in [-1, 0, 1.5]:
        with pytest.raises(ValueError):
            checks.check_stride(stride, dimension_names, shape, "sliding")
