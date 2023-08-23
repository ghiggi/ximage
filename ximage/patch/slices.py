#!/usr/bin/env python3
"""
Created on Sat Dec 10 18:46:00 2022

@author: ghiggi
"""
import numpy as np


def get_slice_size(slc):
    """Get size of the slice.

    Note: The actual slice size must not be representative of the true slice if
    slice.stop is larger than the length of object to be sliced.
    """
    if not isinstance(slc, slice):
        raise TypeError("Expecting slice object")
    size = slc.stop - slc.start
    return size


def pad_slice(slc, padding, min_start=0, max_stop=np.inf):
    """
    Increase/decrease the slice with the padding argument.

    Does not ensure that all output slices have same size.

    Parameters
    ----------
    slc : slice
        Slice objects.
    padding : int
        Padding to be applied to the slice.
    min_start : int, optional
       The minimum value for the start of the new slice.
       The default is 0.
    max_stop : int
        The maximum value for the stop of the new slice.
        The default is np.inf.
    Returns
    -------
    list_slices : TYPE
        The list of slices after applying padding.
    """

    new_slice = slice(max(slc.start - padding, min_start), min(slc.stop + padding, max_stop))
    return new_slice


def pad_slices(list_slices, padding, valid_shape):
    """
    Increase/decrease the list of slices with the padding argument.

    Parameters
    ----------
    list_slices : list
        List of slice objects.
    padding : (int or tuple)
        Padding to be applied on each slice.
    valid_shape : (int or tuple)
        The shape of the array which the slices should be valid on.

    Returns
    -------
    list_slices : TYPE
        The list of slices after applying padding.
    """
    # Check the inputs
    if isinstance(padding, int):
        padding = [padding] * len(list_slices)
    if isinstance(valid_shape, int):
        valid_shape = [valid_shape] * len(list_slices)
    if isinstance(padding, (list, tuple)) and len(padding) != len(list_slices):
        raise ValueError(
            "Invalid padding. The length of padding should be the same as the length of list_slices."
        )
    if isinstance(valid_shape, (list, tuple)) and len(valid_shape) != len(list_slices):
        raise ValueError(
            "Invalid valid_shape. The length of valid_shape should be the same as the length of list_slices."
        )
    # Apply padding
    list_slices = [
        pad_slice(s, padding=p, min_start=0, max_stop=valid_shape[i])
        for i, (s, p) in enumerate(zip(list_slices, padding))
    ]
    return list_slices


# min_size = 10
# min_start = 0
# max_stop = 20
# slc = slice(1, 5)   # left bound
# slc = slice(15, 20) # right bound
# slc = slice(8, 12) # middle


def enlarge_slice(slc, min_size, min_start=0, max_stop=np.inf):
    """
    Enlarge a slice object to have at least a size of min_size.

    The function enforces the left and right bounds of the slice by max_stop and min_start.
    If the original slice size is larger than min_size, the original slice will be returned.

    Parameters
    ----------
    slc : slice
        The original slice object to be enlarged.
    min_size : min_size
        The desired minimum size of the new slice.
    min_start : int, optional
       The minimum value for the start of the new slice.
       The default is 0.
    max_stop : int
        The maximum value for the stop of the new slice.
        The default is np.inf.
    Returns
    -------
    slice
        The new slice object with a size of at least min_size and respecting the left and right bounds.
        If the original slice object is already larger than min_size, the original slice is returned.

    """
    # Get slice size
    slice_size = get_slice_size(slc)

    # If min_size is larger than allowable size, raise error
    if min_size > (max_stop - min_start):
        raise ValueError(
            f"'min_size' {min_size} is too large to generate a slice between {min_start} and {max_stop}."
        )

    # If slice size larger than min_size, return the slice
    if slice_size >= min_size:
        return slc

    # Calculate the number of points to add on both sides
    n_indices_to_add = min_size - slice_size
    add_to_left = add_to_right = n_indices_to_add // 2

    # If n_indices_to_add is odd, add + 1 on the left
    if n_indices_to_add % 2 == 1:
        add_to_left += 1

    # Adjust adding for left and right bounds
    naive_start = slc.start - add_to_left
    naive_stop = slc.stop + add_to_right
    if naive_start <= min_start:
        exceeding_left_size = min_start - naive_start
        add_to_right += exceeding_left_size
        add_to_left -= exceeding_left_size
    if naive_stop >= max_stop:
        exceeding_right_size = naive_stop - max_stop
        add_to_right -= exceeding_right_size
        add_to_left += exceeding_right_size

    # Define new slice
    start = slc.start - add_to_left
    stop = slc.stop + add_to_right
    new_slice = slice(start, stop)

    # Check
    assert get_slice_size(new_slice) == min_size

    # Return new slice
    return new_slice


def enlarge_slices(list_slices, min_size, valid_shape):
    """
    Enlarge a list of slice object to have at least a size of min_size.

    The function enforces the left and right bounds of the slice to be between 0 and valid_shape.
    If the original slice size is larger than min_size, the original slice will be returned.

    Parameters
    ----------
    list_slices : list
        List of slice objects.
    min_size : (int or tuple)
        Minimum size of the output slice.
    valid_shape : (int or tuple)
        The shape of the array which the slices should be valid on.

    Returns
    -------
    list_slices : list
        The list of slices after enlarging it (if necessary).
    """
    # Check the inputs
    if isinstance(min_size, int):
        min_size = [min_size] * len(list_slices)
    if isinstance(valid_shape, int):
        valid_shape = [valid_shape] * len(list_slices)
    if isinstance(min_size, (list, tuple)) and len(min_size) != len(list_slices):
        raise ValueError(
            "Invalid min_size. The length of min_size should be the same as the length of list_slices."
        )
    if isinstance(valid_shape, (list, tuple)) and len(valid_shape) != len(list_slices):
        raise ValueError(
            "Invalid valid_shape. The length of valid_shape should be the same as the length of list_slices."
        )
    # Enlarge the slice
    list_slices = [
        enlarge_slice(slc, min_size=s, min_start=0, max_stop=valid_shape[i])
        for i, (slc, s) in enumerate(zip(list_slices, min_size))
    ]
    return list_slices


def get_slice_from_idx_bounds(idx_start, idx_end):
    """Return the slice required to include the idx bounds."""
    return slice(idx_start, idx_end + 1)


def get_slice_around_index(index, size, min_start=0, max_stop=np.inf):
    """
    Get a slice object of `size` around `index` value.

    If size is larger than (max_stop-min_start), raise an error.

    Parameters
    ----------
    index : int
        The index value around which to retrieve the slice.
    size : int
        The desired size of the slice around the index.
    min_start : int, optional
       The default is np.inf.
       The minimum value for the start of the new slice.
       The default is 0.
    max_stop : int
        The maximum value for the stop of the new slice.

    Returns
    -------
    slice
        A slice object with the desired size and respecting the left and right bounds.

    """

    index_slc = slice(index, index + 1)
    try:
        slc = enlarge_slice(index_slc, min_size=size, min_start=min_start, max_stop=max_stop)
    except ValueError:
        print(index, size, min_start, max_stop, index_slc)
        raise ValueError("'size' {size} is to large to be between {min_start} and {max_stop}.")
    return slc


def _check_buffer(buffer, slice_size):
    if buffer < 0:
        if abs(buffer) * 2 >= slice_size:
            raise ValueError(
                "The negative buffer absolute value is larger than half of the slice_size."
            )
    return buffer


def _check_slice_size(slice_size):
    if slice_size <= 0:
        raise ValueError("slice_size must be a positive non-zero integer.")
    return slice_size


def _check_method(method):
    if not isinstance(method, str):
        raise TypeError("'method' must be a string.")
    valid_methods = ["sliding", "tiling"]
    if method not in valid_methods:
        raise ValueError(f"The only valid 'method' are {valid_methods}.")
    return method


def _check_min_start(min_start, start):
    if min_start is None:
        min_start = start
    if min_start > start:
        raise ValueError("'min_start' can not be larger than 'start'.")
    return min_start


def _check_max_stop(max_stop, stop):
    if max_stop is None:
        max_stop = stop
    if max_stop < stop:
        raise ValueError("'max_stop' can not be smaller than 'stop'.")
    return max_stop


def _check_stride(stride, method):
    if method == "sliding":
        if stride is None:
            stride = 1
        if stride < 1:
            raise ValueError("When sliding, 'stride' must be equal or larger than 1.")
    else:  # tiling
        if stride is None:
            stride = 0
    if not isinstance(stride, int):
        raise TypeError("'stride' must be an integer.")
    return stride


def _get_partitioning_idxs(start, stop, stride, slice_size, method):
    if method == "tiling":
        steps = slice_size + stride
    else:  # sliding
        steps = stride
    idxs = np.arange(start, stop + 1, steps)
    return idxs


def get_partitions_slices(
    start,
    stop,
    slice_size,
    method,
    stride=None,
    buffer=0,
    include_last=True,
    ensure_slice_size=False,
    min_start=None,
    max_stop=None,
):
    """
    Create 1D partitioning list of slices.

    Parameters
    ----------
    start : int
        Slice start.
    stop : int
        slice stop.
    slice_size : int
        Slice size.
    method : str
        Whether to retrieve 'tiling' or 'sliding' slices.
        If 'tiling', start slices are separated by stride+slice_size
        If 'sliding', start slices are separated by stride.
    stride : int, optional
        Step size between slices.
        When 'tiling', the default is 0
        When 'sliding', the default is 1.
        When 'tiling', a positive stride make slices to not overlap and not touch,
        while a negative stride make slices to overlap by 'stride' amount. If stride is 0,
        the slices are contiguous (touch).
        When 'sliding', only a positive stride (>= 1) is allowed.
    buffer:
        The default is 0.
        Value by which to enlarge a slice on each side.
        If stride=0 and buffer is positive, it corresponds to
        the amount of overlap between each tile.
        The final slice size should be slice_size + buffer.
        Depending on min_start and max_stop values, buffering might cause
        border slices to not have same sizes.
    include_last : bool, optional
        Whether to include the last slice if not match slice_size.
        The default is True.
    ensure_slice_size : False, optional
        Used only if include_last is True.
        If False, the last slice does not have size 'slice_size'.
        If True,  the last slice is enlarged to have 'slice_size', by
        tentatively expanded the slice on both sides (accounting for min_start and max_stop).
    min_start: int, optional
        The minimum value that the slices start value can have (after i.e. buffering).
        If None (the default), assumed to be equal to start.
    max_stop: int, optional
        Maximum value that the slices stop value can have (after i.e. buffering).
        If None (the default), assumed to be equal to stop.

    Returns
    -------
    slices : list
        List of slices.

    """
    # Check arguments
    slice_size = _check_slice_size(slice_size)
    method = _check_method(method)
    stride = _check_stride(stride, method)
    buffer = _check_buffer(buffer, slice_size)
    min_start = _check_min_start(min_start, start)
    max_stop = _check_max_stop(max_stop, stop)

    # Define slices
    slice_step = 1  # TODO: modify for dilation together with slice_size
    idxs = _get_partitioning_idxs(
        start=start, stop=stop, stride=stride, slice_size=slice_size, method=method
    )
    slices = [slice(idxs[i], idxs[i] + slice_size, slice_step) for i in range(len(idxs) - 1)]

    # Define last slice
    if include_last and idxs[-1] != stop:
        last_slice = slice(idxs[-1], stop)
        if ensure_slice_size:
            last_slice = enlarge_slice(
                last_slice, min_size=slice_size, min_start=min_start, max_stop=max_stop
            )
        slices.append(last_slice)

    # Buffer the slices
    slices = [
        pad_slice(slc, padding=buffer, min_start=min_start, max_stop=max_stop) for slc in slices
    ]

    return slices


def get_nd_partitions_list_slices(
    list_slices, arr_shape, method, kernel_size, stride, buffer, include_last, ensure_slice_size
):
    """Return the n-dimensional partitions list of slices of a initial list of slices."""
    import itertools

    l_iterables = []
    for i in range(len(list_slices)):
        slice_size = kernel_size[i]
        max_stop = arr_shape[i]
        slc = list_slices[i]
        start = slc.start
        stop = slc.stop
        slices = get_partitions_slices(
            start=start,
            stop=stop,
            slice_size=slice_size,
            method=method,
            stride=stride[i],
            buffer=buffer[i],
            include_last=include_last,
            ensure_slice_size=ensure_slice_size,
            min_start=0,
            max_stop=max_stop,
        )
        l_iterables.append(slices)

    tiles_list_slices = list(itertools.product(*l_iterables))
    return tiles_list_slices
