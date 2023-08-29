#!/usr/bin/env python3
"""
Created on Mon Jul 10 14:07:16 2023

@author: ghiggi
"""
import numpy as np

from ximage.utils.checks import are_all_integers, are_all_natural_numbers


def _ensure_is_dict_argument(arg, dims, arg_name):
    """Ensure argument is a dictionary with same order as dims."""
    if isinstance(arg, (int, float)):
        arg = {dim: arg for dim in dims}
    if isinstance(arg, (list, tuple)):
        if len(arg) != len(dims):
            raise ValueError(f"{arg_name} must match the number of dimensions of the label array.")
        arg = dict(zip(dims, arg))
    if isinstance(arg, dict):
        dict_dims = np.array(list(arg))
        invalid_dims = dict_dims[np.isin(dict_dims, dims, invert=True)].tolist()
        if len(invalid_dims) > 0:
            raise ValueError(
                f"{arg_name} must not contain dimensions {invalid_dims}. It expects only {dims}."
            )
        missing_dims = np.array(dims)[np.isin(dims, dict_dims, invert=True)].tolist()
        if len(missing_dims) > 0:
            raise ValueError(f"{arg_name} must contain also dimensions {missing_dims}")
    else:
        type_str = type(arg)
        raise TypeError(f"Unrecognized type {type_str} for argument {arg_name}.")
    # Reorder as function of dims
    arg = {dim: arg[dim] for dim in dims}
    return arg


def _replace_full_dimension_flag_value(arg, shape):
    """Replace -1 values with the corresponding dimension shape."""
    arg = {dim: shape[i] if value == -1 else value for i, (dim, value) in enumerate(arg.items())}
    return arg


def check_patch_size(patch_size, dims, shape):
    """
    Check the validity of the patch_size argument based on the array shape.

    Parameters
    ----------
    patch_size : (int, list, tuple, dict)
        The size of the patch to extract from the array.
        If int, the patch is a hypercube of size patch_size across all dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have as keys all array dimensions.
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
    for dim, value in patch_size.items():
        if not are_all_natural_numbers(value):
            raise ValueError(
                "Invalid 'patch_size' values. They must be only positive integer values."
            )
    # Check patch size is smaller than array shape
    idx_valid = [value <= max_value for value, max_value in zip(patch_size.values(), shape)]
    max_allowed_patch_size = {dim: value for dim, value in zip(dims, shape)}
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
        If int or float, the kernel is a hypercube of size patch_size across all dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
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
    for dim, value in kernel_size.items():
        if not are_all_natural_numbers(value):
            raise ValueError(
                "Invalid 'kernel_size' values. They must be only positive integer values."
            )
    # Check patch size is smaller than array shape
    idx_valid = [value <= max_value for value, max_value in zip(kernel_size.values(), shape)]
    max_allowed_kernel_size = {dim: value for dim, value in zip(dims, shape)}
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed patch_size values are {max_allowed_kernel_size}")
    return kernel_size


def check_buffer(buffer, dims, shape):
    """
    Check the validity of the buffer argument based on the array shape.

    Parameters
    ----------
    buffer : (int, float, list, tuple or dict)
        The size of the buffer to apply to the array.
        If int or float, equal buffer is set on each dimension of the array.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
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
    for dim, value in buffer.items():
        if not are_all_integers(value):
            raise ValueError("Invalid 'buffer' values. They must be only integer values.")
    return buffer


def check_padding(padding, dims, shape):
    """
    Check the validity of the padding argument based on the array shape.

    Parameters
    ----------
    padding : (int, float, list, tuple or dict)
        The size of the padding to apply to the array.
        If None, zero padding is assumed.
        If int or float, equal padding is set on each dimension of the array.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
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
    for dim, value in padding.items():
        if not are_all_integers(value):
            raise ValueError("Invalid 'padding' values. They must be only integer values.")
    return padding


def check_partitioning_method(partitioning_method):
    """Check partitioning method."""
    if not isinstance(partitioning_method, (str, type(None))):
        raise TypeError("'partitioning_method' must be either a string or None.")
    if isinstance(partitioning_method, str):
        valid_methods = ["sliding", "tiling"]
        if partitioning_method not in valid_methods:
            raise ValueError(f"Valid 'partitioning_method' are {valid_methods}")
    return partitioning_method


def check_stride(stride, dims, shape, partitioning_method):
    """
    Check the validity of the stride argument based on the array shape.

    Parameters
    ----------
    stride : (None, int, float, list, tuple, dict)
        The size of the stride to apply to the array.
        If None, no striding is assumed.
        If int or float, equal stride is set on each dimension of the array.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
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
        if partitioning_method == "tiling":
            stride = 0
        else:  # sliding
            stride = 1
    stride = _ensure_is_dict_argument(stride, dims=dims, arg_name="stride")
    if partitioning_method == "tiling":
        for dim, value in stride.items():
            if not are_all_integers(value):
                raise ValueError("Invalid 'stride' values. They must be only integer values.")
    else:  # sliding
        for dim, value in stride.items():
            if not are_all_natural_numbers(value):
                raise ValueError(
                    "Invalid 'stride' values. They must be only positive integer (>=1) values."
                )
    return stride
