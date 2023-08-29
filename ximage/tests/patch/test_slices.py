import pytest

from ximage.patch import slices


# Tests for public functions ###################################################


def test_get_slice_size():
    """Test get_slice_size"""

    with pytest.raises(TypeError):
        slices.get_slice_size("invalid")

    test_slice = slice(1, 10, 2)
    assert slices.get_slice_size(test_slice) == 9


def test_pad_slice():
    """Test pad_slice"""

    # Always step = 1

    test_slice = slice(2, 8)
    pad = 2
    expected_slice = slice(0, 10)
    assert slices.pad_slice(test_slice, pad) == expected_slice

    # With min and max set
    min_start = 1
    max_stop = 9
    expected_slice = slice(1, 9)
    assert slices.pad_slice(test_slice, pad, min_start, max_stop) == expected_slice


def test_pad_slices():
    """Test pad_slices"""

    test_slices = [slice(2, 8), slice(1, 9)]

    # Integer padding and shape
    pad = 2
    shape = 20
    expected_slices = [slice(0, 10), slice(0, 11)]
    assert slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Truncating with integer shape
    shape = 10
    expected_slices = [slice(0, 10), slice(0, 10)]
    assert slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Tuple padding
    pad = (2, 3)
    shape = 20
    expected_slices = [slice(0, 10), slice(0, 12)]
    assert slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Tuple shape
    pad = 2
    shape = (9, 10)
    expected_slices = [slice(0, 9), slice(0, 10)]
    assert slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Invalid tuple sizes
    pad = (2, 3, 4)
    shape = 10
    with pytest.raises(ValueError):
        slices.pad_slices(test_slices, pad, shape)

    pad = 2
    shape = (9, 10, 11)
    with pytest.raises(ValueError):
        slices.pad_slices(test_slices, pad, shape)


def test_enlarge_slice():
    """Test enlarge_slice"""

    test_slice = slice(3, 5)
    min_start = 1
    max_stop = 10

    # No change
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    min_size = 1
    expected_slice = slice(3, 5)
    assert slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge one side
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    #   |-----|
    min_size = 3
    expected_slice = slice(2, 5)
    assert slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge both sides
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    #   |-------|
    min_size = 4
    expected_slice = slice(2, 6)
    assert slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge reaching min_start
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    # |---------------|
    min_size = 8
    expected_slice = slice(1, 9)
    assert slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge reaching max_stop
    # 1 2 3 4 5 6 7 8 9 10
    #           |---|
    #   |---------------|
    test_slice = slice(6, 8)
    expected_slice = slice(2, 10)
    assert slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge too much
    min_size = 20
    with pytest.raises(ValueError):
        slices.enlarge_slice(test_slice, min_size, min_start, max_stop)


def test_enlarge_slices():
    """Test enlarge_slices"""

    test_slices = [slice(3, 5), slice(6, 8)]

    # Integer min_size and shape
    min_size = 4
    shape = 10
    expected_slices = [slice(2, 6), slice(5, 9)]
    assert slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Capping with integer shape
    shape = 8
    expected_slices = [slice(2, 6), slice(4, 8)]
    assert slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Tuple min_size
    min_size = (4, 6)
    shape = 10
    expected_slices = [slice(2, 6), slice(4, 10)]
    assert slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Tuple shape
    min_size = 4
    shape = (5, 8)
    expected_slices = [slice(1, 5), slice(4, 8)]
    assert slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Invalid tuple sizes
    min_size = (4, 5, 6)
    shape = 10
    with pytest.raises(ValueError):
        slices.enlarge_slices(test_slices, min_size, shape)

    min_size = 4
    shape = (8, 9, 10)
    with pytest.raises(ValueError):
        slices.enlarge_slices(test_slices, min_size, shape)


def test_get_slice_from_idx_bounds():
    """Test get_slice_from_idx_bounds"""

    idx_start = 2
    idx_stop = 8
    expected_slice = slice(2, 9)
    assert slices.get_slice_from_idx_bounds(idx_start, idx_stop) == expected_slice


def test_get_slice_around_index():
    """Test get_slice_around_index"""

    index = 3
    min_start = 1
    max_stop = 10

    # Size 1
    # 1 2 3 4 5 6 7 8 9 10
    #     *-|
    size = 1
    expected_slice = slice(3, 4)
    assert slices.get_slice_around_index(index, size, min_start, max_stop) == expected_slice

    # Even size
    # 1 2 3 4 5 6 7 8 9 10
    #   |-*-|
    size = 2
    expected_slice = slice(2, 4)
    assert slices.get_slice_around_index(index, size, min_start, max_stop) == expected_slice

    # Odd size
    # 1 2 3 4 5 6 7 8 9 10
    #   |-*---|
    size = 3
    expected_slice = slice(2, 5)
    assert slices.get_slice_around_index(index, size, min_start, max_stop) == expected_slice

    # Reaching min_start
    # 1 2 3 4 5 6 7 8 9 10
    # |---*-------|
    size = 6
    expected_slice = slice(1, 7)
    assert slices.get_slice_around_index(index, size, min_start, max_stop) == expected_slice

    # Too large
    size = 20
    with pytest.raises(ValueError):
        slices.get_slice_around_index(index, size, min_start, max_stop)


def test_get_partitions_slices():
    """Test get_partitions_slices"""

    start = 0
    stop = 10
    slice_size = 3
    method = "tiling"

    # Default (with tiling)
    # 0 1 2 3 4 5 6 7 8 9 10
    # |-----|-----|-----|-|
    expected_slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 10)]
    assert slices.get_partitions_slices(start, stop, slice_size, method) == expected_slices

    # Stride, positive
    # 0 1 2 3 4 5 6 7 8 9 10
    # |-----| |-----| |---|
    stride = 1
    expected_slices = [slice(0, 3), slice(4, 7), slice(8, 10)]
    assert (
        slices.get_partitions_slices(start, stop, slice_size, method, stride=stride)
        == expected_slices
    )

    # Stride, negative
    # 0 1 2 3 4 5 6 7 8 9 10
    # |-----| |-----| |---|
    #     |-----| |-----|
    stride = -1
    expected_slices = [slice(0, 3), slice(2, 5), slice(4, 7), slice(6, 9), slice(8, 10)]
    assert (
        slices.get_partitions_slices(start, stop, slice_size, method, stride=stride)
        == expected_slices
    )

    # Buffer (extending)
    # 0 1 2 3 4 5 6 7 8 9 10
    # |----->-| |-<----->-|
    #     |-<----->-| |-<-|
    buffer = 1
    expected_slices = [slice(0, 4), slice(2, 7), slice(5, 10), slice(8, 10)]
    assert (
        slices.get_partitions_slices(start, stop, slice_size, method, buffer=buffer)
        == expected_slices
    )

    # Remove last
    # 0 1 2 3 4 5 6 7 8 9 10
    # |-----|-----|-----|
    expected_slices = [slice(0, 3), slice(3, 6), slice(6, 9)]
    assert (
        slices.get_partitions_slices(start, stop, slice_size, method, include_last=False)
        == expected_slices
    )

    # Resize last
    # 0 1 2 3 4 5 6 7 8 9 10
    # |-----|-----|-¦===|-|
    expected_slices = [slice(0, 3), slice(3, 6), slice(6, 9), slice(7, 10)]
    assert (
        slices.get_partitions_slices(start, stop, slice_size, method, ensure_slice_size=True)
        == expected_slices
    )

    # min_start and max_stop different from start and stop with buffering
    #  0 1 2 3 4 5 6 7 8 9 1011
    #  |-<----->-| |-<----->-|
    #        |-<----->-| |-<->-|
    buffer = 1
    min_start = -1
    max_stop = 11
    expected_slices = [slice(-1, 4), slice(2, 7), slice(5, 10), slice(8, 11)]
    assert (
        slices.get_partitions_slices(
            start, stop, slice_size, method, buffer=buffer, min_start=min_start, max_stop=max_stop
        )
        == expected_slices
    )

    # Sliding, with stride
    # 0 1 2 3 4 5 6 7 8 9 10
    # |-----| |-----| |---|
    slice_size = 3
    method = "sliding"
    stride = 4
    expected_slices = [slice(0, 3), slice(4, 7), slice(8, 10)]
    assert (
        slices.get_partitions_slices(start, stop, slice_size, method, stride=stride)
        == expected_slices
    )

    # Sliding, invalid stride
    stride = 0
    with pytest.raises(ValueError):
        slices.get_partitions_slices(start, stop, slice_size, method, stride=stride)


def test_get_nd_partitions_list_slices():
    """Test get_nd_partitions_list_slices"""

    starts = [0, 10]
    stops = [10, 20]
    intervals = [slice(starts[i], stops[i]) for i in range(len(starts))]
    slice_sizes = [3, 3]
    shape = (20, 20)
    strides = [1, 0]
    buffers = [0, 1]

    # Base slices
    # 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 20
    # |-----| |-----| |---|
    #                   |-----|===|-|===|-|===|
    base_slices1 = [slice(0, 3), slice(4, 7), slice(8, 10)]
    base_slices2 = [slice(9, 14), slice(12, 17), slice(15, 20), slice(18, 20)]

    expected_slices = []  # list of each pair
    for base_slice1 in base_slices1:
        for base_slice2 in base_slices2:
            expected_slices.append((base_slice1, base_slice2))

    result = slices.get_nd_partitions_list_slices(
        intervals, shape, "tiling", slice_sizes, strides, buffers, True, False
    )

    assert result == expected_slices


# Tests for internal functions #################################################


def test_check_buffer():
    """Test _check_buffer"""

    # For slice(start, end) to make sense, end must be at least (start + 1)

    slice_size = 3

    # Valid buffer
    # 0 1 2 3
    # |-----|
    #   |-|
    buffer = -1
    assert slices._check_buffer(buffer, slice_size) == buffer

    # Invalid buffer
    # 0 1 2 3
    # |-----|
    #    .
    buffer = -2
    with pytest.raises(ValueError):
        slices._check_buffer(buffer, slice_size)

    slice_size = 4

    # Valid buffer
    # 0 1 2 3 4
    # |-------|
    #   |---|
    buffer = -1
    assert slices._check_buffer(buffer, slice_size) == buffer

    # Invalid buffer
    # 0 1 2 3 4
    # |-------|
    #     |
    buffer = -2
    with pytest.raises(ValueError):
        slices._check_buffer(buffer, slice_size)


def test_check_slice_size():
    """Test _check_slice_size"""

    # Valid slice_size
    slice_size = 1
    assert slices._check_slice_size(slice_size) == slice_size

    # Invalid slice_size
    for slice_size in [0, -1]:
        with pytest.raises(ValueError):
            slices._check_slice_size(slice_size)


def test_check_method():
    """Test _check_method"""

    valid_methods = ["sliding", "tiling"]

    for method in valid_methods:
        assert slices._check_method(method) == method

    # Invalid methods
    with pytest.raises(ValueError):
        slices._check_method("invalid")

    with pytest.raises(TypeError):
        slices._check_method(1)


def test_check_min_start():
    """Test _check_min_start"""

    start = 10

    # min_start not provided
    assert slices._check_min_start(None, start) == start

    # Valid min_start
    valid_min_starts = [9, 10]
    for min_start in valid_min_starts:
        assert slices._check_min_start(min_start, start) == min_start

    # Invalid min_start
    invalid_min_start = 11
    with pytest.raises(ValueError):
        slices._check_min_start(invalid_min_start, start)


def test_check_max_stop():
    """Test _check_max_stop"""

    stop = 10

    # max_stop not provided
    assert slices._check_max_stop(None, stop) == stop

    # Valid max_stop
    valid_max_stops = [10, 11]
    for max_stop in valid_max_stops:
        assert slices._check_max_stop(max_stop, stop) == max_stop

    # Invalid max_stop
    invalid_max_stop = 9
    with pytest.raises(ValueError):
        slices._check_max_stop(invalid_max_stop, stop)


def test_check_stride():
    """Test _check_stride"""

    # Sliding method
    method = "sliding"

    # Default stride
    assert slices._check_stride(None, method) == 1

    # Valid strides
    valid_strides = [1, 2]
    for stride in valid_strides:
        assert slices._check_stride(stride, method) == stride

    # Invalid strides
    invalid_strides = [0, -1]
    for stride in invalid_strides:
        with pytest.raises(ValueError):
            slices._check_stride(stride, method)

    invalid_strides = [1.0, 1.5, "1"]
    for stride in invalid_strides:
        with pytest.raises(TypeError):
            slices._check_stride(stride, method)

    # Tiling method
    method = "tiling"

    # Default stride
    assert slices._check_stride(None, method) == 0

    # Valid strides
    valid_strides = [-1, 0, 1]
    for stride in valid_strides:
        assert slices._check_stride(stride, method) == stride

    # Invalid stride
    invalid_strides = [0.0, 1.5, "1"]
    for stride in invalid_strides:
        with pytest.raises(TypeError):
            slices._check_stride(stride, method)
