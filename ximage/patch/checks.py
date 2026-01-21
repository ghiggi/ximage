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
"""Checks for patch extraction function arguments."""
import numpy as np

from ximage.utils.checks import are_all_integers, are_all_natural_numbers


def _ensure_is_dict_argument(arg, dims, arg_name):
    """Ensure argument is a dictionary with same order as dims."""
    if isinstance(arg, (int, float)):
        arg = dict.fromkeys(dims, arg)
    if isinstance(arg, (list, tuple)):
        if len(arg) != len(dims):
            raise ValueError(f"{arg_name} must match the number of dimensions of the label array.")
        arg = dict(zip(dims, arg, strict=True))
    if isinstance(arg, dict):
        dict_dims = np.array(list(arg))
        invalid_dims = dict_dims[np.isin(dict_dims, dims, invert=True)].tolist()
        if len(invalid_dims) > 0:
            raise ValueError(f"{arg_name} must not contain dimensions {invalid_dims}. It expects only {dims}.")
        missing_dims = np.array(dims)[np.isin(dims, dict_dims, invert=True)].tolist()
        if len(missing_dims) > 0:
            raise ValueError(f"{arg_name} must contain also dimensions {missing_dims}")
    else:
        type_str = type(arg)
        raise TypeError(f"Unrecognized type {type_str} for argument {arg_name}.")
    # Reorder arguments as function of dims
    return {dim: arg[dim] for dim in dims}


def _replace_full_dimension_flag_value(arg, shape):
    """Replace -1 values with the corresponding dimension shape."""
    # Return argument with positive integer values
    return {dim: shape[i] if value == -1 else value for i, (dim, value) in enumerate(arg.items())}


def check_patch_size(patch_size, dims, shape):
    """
    Check the validity of the ``patch_size`` argument based on the array shape.

    Parameters
    ----------
    patch_size : (int, list, tuple, dict)
        The size of the patch to extract from the array.
        If int, the patch is a hypercube of size patch_size across all dimensions.
        If ``list`` or ``tuple``, the length must match the number of dimensions of the array.
        If a ``dict``, it must have as keys all array dimensions.
        The value -1 can be used to specify the full array dimension shape.
        Otherwise, only positive integers values (>1) are accepted.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    patch_size : dict
        The shape of the patch.
    """
    patch_size = _ensure_is_dict_argument(patch_size, dims=dims, arg_name="patch_size")
    patch_size = _replace_full_dimension_flag_value(patch_size, shape)
    # Check natural number
    for value in patch_size.values():
        if not are_all_natural_numbers(value):
            raise ValueError("Invalid 'patch_size' values. They must be only positive integer values.")
    # Check patch size is smaller than array shape
    idx_valid = [value <= max_value for value, max_value in zip(patch_size.values(), shape, strict=True)]
    max_allowed_patch_size = dict(zip(dims, shape, strict=True))
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed patch_size values are {max_allowed_patch_size}")
    return patch_size


def check_kernel_size(kernel_size, dims, shape):
    """
    Check the validity of the kernel_size argument based on the array shape.

    Parameters
    ----------
    kernel_size : (int, list, tuple, dict)
        The size of the kernel to extract from the array.
        If ``int`` or ``float``, the kernel is a hypercube of size patch_size across all dimensions.
        If ``list`` or ``tuple``, the length must match the number of dimensions of the array.
        If a ``dict``, it must have has keys all array dimensions.
        The value -1 can be used to specify the full array dimension shape.
        Otherwise, only positive integers values (>1) are accepted.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    kernel_size : dict
        The shape of the kernel.
    """
    kernel_size = _ensure_is_dict_argument(kernel_size, dims=dims, arg_name="kernel_size")
    kernel_size = _replace_full_dimension_flag_value(kernel_size, shape)
    # Check natural number
    for value in kernel_size.values():
        if not are_all_natural_numbers(value):
            raise ValueError("Invalid 'kernel_size' values. They must be only positive integer values.")
    # Check patch size is smaller than array shape
    idx_valid = [value <= max_value for value, max_value in zip(kernel_size.values(), shape, strict=True)]
    max_allowed_kernel_size = dict(zip(dims, shape, strict=True))
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed patch_size values are {max_allowed_kernel_size}.")
    return kernel_size


def check_buffer(buffer, dims, shape):
    """
    Check the validity of the buffer argument based on the array shape.

    Parameters
    ----------
    buffer : (int, float, list, tuple or dict)
        The size of the buffer to apply to the array.
        If ``int`` or ``float``, equal buffer is set on each dimension of the array.
        If ``list`` or ``tuple``, the length must match the number of dimensions of the array.
        If a ``dict``, it must have has keys all array dimensions.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    buffer : dict
        The buffer to apply on each dimension.
    """
    buffer = _ensure_is_dict_argument(buffer, dims=dims, arg_name="buffer")
    for value in buffer.values():
        if not are_all_integers(value):
            raise ValueError("Invalid 'buffer' values. They must be only integer values.")
    # Check buffer is smaller than half the array shape
    dict_max_values = {dim: int(np.floor(size / 2)) for dim, size in zip(buffer.keys(), shape, strict=True)}
    idx_valid = [value <= dict_max_values[dim] for dim, value in buffer.items()]
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed 'buffer' values are {dict_max_values}.")
    return buffer


def check_padding(padding, dims, shape):
    """
    Check the validity of the padding argument based on the array shape.

    Parameters
    ----------
    padding : (int, float, list, tuple or dict)
        The size of the padding to apply to the array.
        If ``None``, zero padding is assumed.
        If ``int`` or ``float``, equal padding is set on each dimension of the array.
        If ``list`` or ``tuple``, the length must match the number of dimensions of the array.
        If a ``dict``, it must have has keys all array dimensions.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    padding : dict
        The padding to apply on each dimension.
    """
    padding = _ensure_is_dict_argument(padding, dims=dims, arg_name="padding")
    for value in padding.values():
        if not are_all_integers(value):
            raise ValueError("Invalid 'padding' values. They must be only integer values.")
    # Check padding is smaller than half the array shape
    dict_max_values = {dim: int(np.floor(size / 2)) for dim, size in zip(padding.keys(), shape, strict=True)}
    idx_valid = [value <= dict_max_values[dim] for dim, value in padding.items()]
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed 'padding' values are {dict_max_values}.")
    return padding


def check_partitioning_method(partitioning_method):
    """Check partitioning method."""
    if not isinstance(partitioning_method, (str, type(None))):
        raise TypeError("'partitioning_method' must be either a string or None.")
    if isinstance(partitioning_method, str):
        valid_methods = ["sliding", "tiling"]
        if partitioning_method not in valid_methods:
            raise ValueError(f"Valid 'partitioning_method' are {valid_methods}.")
    return partitioning_method


def check_stride(stride, dims, shape, partitioning_method):
    """
    Check the validity of the stride argument based on the array shape.

    Parameters
    ----------
    stride : (None, int, float, list, tuple, dict)
        The size of the stride to apply to the array.
        If None, no striding is assumed.
        If ``int`` or ``float``, equal stride is set on each dimension of the array.
        If ``list`` or ``tuple``, the length must match the number of dimensions of the array.
        If a ``dict``, it must have has keys all array dimensions.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.
    partitioning_method: (None, str)
        The optional partitioning method (tiling or sliding) to use.

    Returns
    -------
    stride : dict
        The stride to apply on each dimension.
    """
    if partitioning_method is None:
        return None
    # Set default arguments
    if stride is None:
        stride = 0 if partitioning_method == "tiling" else 1
    stride = _ensure_is_dict_argument(stride, dims=dims, arg_name="stride")
    # If tiling, check just that are integers
    # --> Negative strides lead to overlapping
    # --> Positive strides lead to not contiguous tiles
    if partitioning_method == "tiling":
        for value in stride.values():
            if not are_all_integers(value):
                raise ValueError("Invalid 'stride' values. They must be only integer values.")
    # If sliding, check are only positive numbers !
    else:  # sliding
        for value in stride.values():
            if not are_all_natural_numbers(value):
                raise ValueError("Invalid 'stride' values. They must be only positive integer (>=1) values.")
    # Check stride values are smaller than half the array shape
    # --> A stride with value exactly equal to half the array shape is equivalent to tiling
    dict_max_values = {dim: int(np.floor(size / 2)) for dim, size in zip(stride.keys(), shape, strict=True)}
    idx_valid = [value <= dict_max_values[dim] for dim, value in stride.items()]
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed 'stride' values are {dict_max_values}.")
    return stride
