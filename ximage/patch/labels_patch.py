#!/usr/bin/env python3
"""
Created on Wed Oct 19 19:40:12 2022

@author: ghiggi
"""
import random
import warnings
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ximage.labels.labels import highlight_label
from ximage.patch.checks import (
    are_all_natural_numbers,
    check_buffer,
    check_kernel_size,
    check_padding,
    check_partitioning_method,
    check_patch_size,
    check_stride,
)
from ximage.patch.plot2d import plot_label_patch_extraction_areas
from ximage.patch.slices import (
    enlarge_slices,
    get_nd_partitions_list_slices,
    get_slice_around_index,
    get_slice_from_idx_bounds,
    pad_slices,
)

# -----------------------------------------------------------------------------.
#### TODOs
## Partitioning
# - Option to bound min_start and max_stop to labels bbox
# - Option to define min_start and max_stop to be divisible by patch_size + stride
# - When tiling ... define start so to center tiles around label_bbox, instead of starting at label bbox start
# - Option: partition_only_when_label_bbox_exceed_patch_size

# - Add option that returns a flag if the point center is the actual identified one,
#   or was close to the boundary !

# -----------------------------------------------------------------------------.
# - Implement dilate option (to subset pixel within partitions).
#   --> slice(start, stop, step=dilate) ... with patch_size redefined at start to patch_size*dilate
#   --> Need updates of enlarge slcies, pad_slices utilities (but first test current usage !)

# -----------------------------------------------------------------------------.

## Image sliding/tiling reconstruction
# - get_index_overlapping_slices
# - trim: bool, keyword only
#   Whether or not to trim stride elements from each block after calling the map function.
#   Set this to False if your mapping function already does this for you.
#   This for when merging !

####--------------------------------------------------------------------------.


def _check_label_arr(label_arr):
    """Check label_arr."""
    # Note: If label array is all zero or nan, labels_id will be []

    # Put label array in memory
    label_arr = np.asanyarray(label_arr)

    # Set 0 label to nan
    label_arr = label_arr.astype(float)  # otherwise if int throw an error when assigning nan
    label_arr[label_arr == 0] = np.nan

    # Check labels_id are natural number >= 1
    valid_labels = np.unique(label_arr[~np.isnan(label_arr)])
    if not are_all_natural_numbers(valid_labels):
        raise ValueError("The label array contains non positive natural numbers.")

    return label_arr


def _check_labels_id(labels_id, label_arr):
    """Check labels_id."""
    # Check labels_id type
    if not isinstance(labels_id, (type(None), int, list, np.ndarray)):
        raise TypeError("labels_id must be None or a list or a np.array.")
    if isinstance(labels_id, int):
        labels_id = [labels_id]
    # Get list of valid labels
    valid_labels = np.unique(label_arr[~np.isnan(label_arr)]).astype(int)
    # If labels_id is None, assign the valid_labels
    if isinstance(labels_id, type(None)):
        labels_id = valid_labels
        return labels_id
    # If input labels_id is a list, make it a np.array
    labels_id = np.array(labels_id).astype(int)
    # Check labels_id are natural number >= 1
    if np.any(labels_id == 0):
        raise ValueError("labels id must not contain the 0 value.")
    if not are_all_natural_numbers(labels_id):
        raise ValueError("labels id must be positive natural numbers.")
    # Check labels_id are number present in the label_arr
    invalid_labels = labels_id[~np.isin(labels_id, valid_labels)]
    if invalid_labels.size != 0:
        invalid_labels = invalid_labels.astype(int)
        raise ValueError(f"The following labels id are not valid: {invalid_labels}")
    # If no labels, no patch to extract
    n_labels = len(labels_id)
    if n_labels == 0:
        raise ValueError("No labels available.")
    return labels_id


def _check_n_patches_per_partition(n_patches_per_partition, centered_on):
    """
    Check the number of patches to extract from each partition.

    It is used only if centered_on is a callable or 'random'

    Parameters
    ----------
    n_patches_per_partition : int
        Number of patches to extract from each partition.
    centered_on : (str, callable)
        Method to extract the patch around a label point.

    Returns
    -------
    n_patches_per_partition: int
       The number of patches to extract from each partition.
    """
    if n_patches_per_partition < 1:
        raise ValueError("n_patches_per_partitions must be a positive integer.")
    if isinstance(centered_on, str):
        if centered_on not in ["random"]:
            if n_patches_per_partition > 1:
                raise ValueError(
                    "Only the pre-implemented centered_on='random' method allow n_patches_per_partition values > 1."
                )
    return n_patches_per_partition


def _check_n_patches_per_label(n_patches_per_label, n_patches_per_partition):
    if n_patches_per_label < n_patches_per_partition:
        raise ValueError("n_patches_per_label must be equal or larger to n_patches_per_partition.")
    return n_patches_per_label


def _check_callable_centered_on(centered_on):
    """Check validity of callable centered_on."""
    input_shape = (2, 3)
    arr = np.zeros(input_shape)
    point = centered_on(arr)
    if not isinstance(point, (tuple, type(None))):
        raise ValueError(
            "The 'centered_on' function should return a point coordinates tuple or None."
        )
    if len(point) != len(input_shape):
        raise ValueError(
            "The 'centered_on' function should return point coordinates having same dimensions has input array."
        )
    for c, max_value in zip(point, input_shape):
        if c < 0:
            raise ValueError("The point coordinate must be a positive integer.")
        if c >= max_value:
            raise ValueError("The point coordinate must be inside the array shape.")
        if np.isnan(c):
            raise ValueError("The point coordinate must not be np.nan.")
    try:
        point = centered_on(arr * np.nan)
        if point is not None:
            raise ValueError(
                "The 'centered_on' function should return None if the input array is a np.nan ndarray."
            )
    except:
        raise ValueError("The 'centered_on' function should be able to deal with a np.nan ndarray.")


def _check_centered_on(centered_on):
    """Check valid centered_on to identify a point in an array."""
    if not (callable(centered_on) or isinstance(centered_on, str)):
        raise TypeError("'centered_on' must be a string or a function.")
    if isinstance(centered_on, str):
        valid_centered_on = [
            "max",
            "min",
            "centroid",
            "center_of_mass",
            "random",
            "label_bbox",  # unfixed patch_size
        ]
        if centered_on not in valid_centered_on:
            raise ValueError(f"Valid 'centered_on' values are: {valid_centered_on}.")

    if callable(centered_on):
        _check_callable_centered_on(centered_on)
    return centered_on


def _get_variable_arr(xr_obj, variable, centered_on):
    """Get variable array (in memory)."""
    if isinstance(xr_obj, xr.DataArray):
        variable_arr = np.asanyarray(xr_obj.data)
        return variable_arr
    else:
        if centered_on is not None:
            if variable is None and (centered_on in ["max", "min"] or callable(centered_on)):
                raise ValueError("'variable' must be specified if 'centered_on' is specified.")
        if variable is not None:
            variable_arr = np.asanyarray(xr_obj[variable].data)  # in memory

        else:
            variable_arr = None
    return variable_arr


def _check_variable_arr(variable_arr, label_arr):
    """Check variable array validity."""
    if variable_arr is not None:
        if variable_arr.shape != label_arr.shape:
            raise ValueError(
                "Arrays corresponding to 'variable' and 'label_name' must have same shape."
            )
    return variable_arr


def _get_point_centroid(arr):
    """Get the coordinate of label bounding box center.

    It assumes that the array has been cropped around the label.
    It returns None if all values are non-finite (i.e. np.nan).
    """
    if np.all(~np.isfinite(arr)):
        return None
    centroid = np.array(arr.shape) / 2.0
    centroid = tuple(centroid.tolist())
    return centroid


def _get_point_random(arr):
    """Get random point with finite value."""
    is_finite = np.isfinite(arr)
    if np.all(~is_finite):
        return None
    points = np.argwhere(is_finite)
    random_point = random.choice(points)
    return random_point


def _get_point_with_max_value(arr):
    """Get point with maximum value."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        point = np.argwhere(arr == np.nanmax(arr))
    if len(point) == 0:
        point = None
    else:
        point = tuple(point[0].tolist())
    return point


def _get_point_with_min_value(arr):
    """Get point with minimum value."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        point = np.argwhere(arr == np.nanmin(arr))
    if len(point) == 0:
        point = None
    else:
        point = tuple(point[0].tolist())
    return point


def _get_point_center_of_mass(arr, integer_index=True):
    """Get the coordinate of the label center of mass.

    It uses all cells which have finite values.
    If 0 value should be a non-label area, mask before with np.nan.
    It returns None if all values are non-finite (i.e. np.nan).
    """
    indices = np.argwhere(np.isfinite(arr))
    if len(indices) == 0:
        return None
    center_of_mass = np.nanmean(indices, axis=0)
    if integer_index:
        center_of_mass = center_of_mass.round().astype(int)
    center_of_mass = tuple(center_of_mass.tolist())
    return center_of_mass


def find_point(arr, centered_on: Union[str, Callable] = "max"):
    """Find a specific point coordinate of the array.

    If the coordinate can't be find, return None.
    """
    centered_on = _check_centered_on(centered_on)

    if centered_on == "max":
        point = _get_point_with_max_value(arr)
    elif centered_on == "min":
        point = _get_point_with_min_value(arr)
    elif centered_on == "centroid":
        point = _get_point_centroid(arr)
    elif centered_on == "center_of_mass":
        point = _get_point_center_of_mass(arr)
    elif centered_on == "random":
        point = _get_point_random(arr)
    else:  # callable centered_on
        point = centered_on(arr)
    if point is not None:
        point = tuple(int(p) for p in point)
    return point


def _get_labels_bbox_slices(arr):
    """
    Compute the bounding box slices of non-zero elements in a n-dimensional numpy array.

    Assume that only one unique non-zero elements values is present in the array.
    Assume that NaN and Inf have been replaced by zeros.

    Other implementations: scipy.ndimage.find_objects

    Parameters
    ----------
    arr : np.ndarray
        n-dimensional numpy array.

    Returns
    -------
    list_slices : list
        List of slices to extract the region with non-zero elements in the input array.
    """
    # Return None if all values are zeros
    if not np.any(arr):
        return None

    ndims = arr.ndim
    coords = np.nonzero(arr)
    list_slices = [
        get_slice_from_idx_bounds(np.min(coords[i]), np.max(coords[i])) for i in range(ndims)
    ]
    return list_slices


def _get_patch_list_slices_around_label_point(
    label_arr,
    label_id,
    variable_arr,
    patch_size,
    centered_on,
):
    """Get list_slices to extract patch around a label point.

    Assume label_arr must match variable_arr shape.
    Assume patch_size shape must match variable_arr shape .
    """
    # Subset variable_arr around label
    list_slices = _get_labels_bbox_slices(label_arr == label_id)
    if list_slices is None:
        return None
    label_subset_arr = label_arr[tuple(list_slices)]
    variable_subset_arr = variable_arr[tuple(list_slices)]
    variable_subset_arr = np.asarray(variable_subset_arr)  # if dask, make numpy
    # Mask variable arr outside the label
    variable_subset_arr[label_subset_arr != label_id] = np.nan
    # Find point of subset array
    point_subset_arr = find_point(arr=variable_subset_arr, centered_on=centered_on)
    # Define patch list_slices
    if point_subset_arr is not None:
        # Find point in original array
        point = [slc.start + c for slc, c in zip(list_slices, point_subset_arr)]
        # Find patch list slices
        patch_list_slices = [
            get_slice_around_index(p, size=size, min_start=0, max_stop=shape)
            for p, size, shape in zip(point, patch_size, variable_arr.shape)
        ]
        # TODO: also return a flag if the p midpoint is conserved (by +/- 1) or not
    else:
        patch_list_slices = None
    return patch_list_slices


def _get_patch_list_slices_around_label(label_arr, label_id, padding, min_patch_size):
    """Get list_slices to extract patch around a label."""
    # Get label bounding box slices
    list_slices = _get_labels_bbox_slices(label_arr == label_id)
    if list_slices is None:
        return None
    # Apply padding to the slices
    list_slices = pad_slices(list_slices, padding=padding, valid_shape=label_arr.shape)
    # Increase slices to match min_patch_size
    list_slices = enlarge_slices(list_slices, min_size=min_patch_size, valid_shape=label_arr.shape)
    return list_slices


def _get_patch_list_slices(label_arr, label_id, variable_arr, patch_size, centered_on, padding):
    """Get patch n-dimensional list slices."""
    if not callable(centered_on) and centered_on == "label_bbox":
        list_slices = _get_patch_list_slices_around_label(
            label_arr=label_arr, label_id=label_id, padding=padding, min_patch_size=patch_size
        )
    else:
        list_slices = _get_patch_list_slices_around_label_point(
            label_arr=label_arr,
            label_id=label_id,
            variable_arr=variable_arr,
            patch_size=patch_size,
            centered_on=centered_on,
        )
    return list_slices


def _get_masked_arrays(label_arr, variable_arr, partition_list_slices):
    """Mask labels and variable arrays outside the partitions area."""
    masked_partition_label_arr = np.zeros(label_arr.shape) * np.nan
    masked_partition_label_arr[tuple(partition_list_slices)] = label_arr[
        tuple(partition_list_slices)
    ]
    if variable_arr is not None:
        masked_partition_variable_arr = np.zeros(variable_arr.shape) * np.nan
        masked_partition_variable_arr[tuple(partition_list_slices)] = variable_arr[
            tuple(partition_list_slices)
        ]
    else:
        masked_partition_variable_arr = None
    return masked_partition_label_arr, masked_partition_variable_arr


def _get_patches_from_partitions_list_slices(
    partitions_list_slices,
    label_arr,
    variable_arr,
    label_id,
    patch_size,
    centered_on,
    n_patches_per_partition,
    padding,
    verbose=False,
):
    """Return patches list slices from list of partitions list_slices.

    n_patches_per_partition is 1 unless centered_on is 'random' or a callable.
    """
    patches_list_slices = []
    for partition_list_slices in partitions_list_slices:
        if verbose:
            print(f" -  partition: {partition_list_slices}")
        masked_label_arr, masked_variable_arr = _get_masked_arrays(
            label_arr=label_arr,
            variable_arr=variable_arr,
            partition_list_slices=partition_list_slices,
        )
        n = 0
        for n in range(n_patches_per_partition):
            patch_list_slices = _get_patch_list_slices(
                label_arr=masked_label_arr,
                variable_arr=masked_variable_arr,
                label_id=label_id,
                patch_size=patch_size,
                centered_on=centered_on,
                padding=padding,
            )
            if patch_list_slices is not None and patch_list_slices not in patches_list_slices:
                n += 1
                patches_list_slices.append(patch_list_slices)
    return patches_list_slices


def _get_list_isel_dicts(patches_list_slices, dims):
    """Return a list with isel dictionaries."""
    list_isel_dicts = []
    for patch_list_slices in patches_list_slices:
        # list_isel_dicts.append(dict(zip(dims, patch_list_slices)))
        list_isel_dicts.append({dim: slc for dim, slc in zip(dims, patch_list_slices)})
    return list_isel_dicts


def _extract_xr_patch(xr_obj, isel_dict, label_name, label_id, highlight_label_id):
    """Extract a xarray patch."""
    # Extract xarray patch around label
    xr_obj_patch = xr_obj.isel(isel_dict)

    # If asked, set label array to 0 except for label_id
    if highlight_label_id:
        xr_obj_patch = highlight_label(xr_obj_patch, label_name=label_name, label_id=label_id)
    return xr_obj_patch


def _get_patches_isel_dict_generator(
    xr_obj,
    label_name,
    patch_size,
    variable=None,
    # Output options
    n_patches=np.Inf,
    n_labels=None,
    labels_id=None,
    grouped_by_labels_id=False,
    # (Tile) label patch extraction
    padding=0,
    centered_on="max",
    n_patches_per_label=np.Inf,
    n_patches_per_partition=1,
    debug=False,
    # Label Tiling/Sliding Options
    partitioning_method=None,
    n_partitions_per_label=None,
    kernel_size=None,
    buffer=0,
    stride=None,
    include_last=True,
    ensure_slice_size=True,
    verbose=False,
):
    # Get label array information
    label_arr = xr_obj[label_name].data
    dims = xr_obj[label_name].dims
    shape = label_arr.shape

    # Check input arguments
    if n_labels is not None and labels_id is not None:
        raise ValueError("Specify either n_labels or labels_id.")
    if kernel_size is None:
        kernel_size = patch_size
    patch_size = check_patch_size(patch_size, dims, shape)
    buffer = check_buffer(buffer, dims, shape)
    padding = check_padding(padding, dims, shape)

    partitioning_method = check_partitioning_method(partitioning_method)
    stride = check_stride(stride, dims, shape, partitioning_method)
    kernel_size = check_kernel_size(kernel_size, dims, shape)

    centered_on = _check_centered_on(centered_on)
    n_patches_per_partition = _check_n_patches_per_partition(n_patches_per_partition, centered_on)
    n_patches_per_label = _check_n_patches_per_label(n_patches_per_label, n_patches_per_partition)

    label_arr = _check_label_arr(label_arr)  # output is np.array !
    labels_id = _check_labels_id(labels_id=labels_id, label_arr=label_arr)
    variable_arr = _get_variable_arr(xr_obj, variable, centered_on)  # if required
    variable_arr = _check_variable_arr(variable_arr, label_arr)

    # Define number of labels from which to extract patches
    available_n_labels = len(labels_id)
    n_labels = min(available_n_labels, n_labels) if n_labels else available_n_labels
    if verbose:
        print(f"Extracting patches from {n_labels} labels.")
    # -------------------------------------------------------------------------.
    # Extract patch(es) around the label
    patch_counter = 0
    break_flag = False
    for i, label_id in enumerate(labels_id[0:n_labels]):
        if verbose:
            print(f"Label ID: {label_id} ({i}/{n_labels})")

        # Subset label_arr around the given label
        label_bbox_slices = _get_labels_bbox_slices(label_arr == label_id)

        # Apply padding to the label bounding box
        label_bbox_slices = pad_slices(
            label_bbox_slices, padding=padding.values(), valid_shape=label_arr.shape
        )

        # --------------------------------------------------------------------.
        # Retrieve partitions list_slices
        if partitioning_method is not None:
            partitions_list_slices = get_nd_partitions_list_slices(
                label_bbox_slices,
                arr_shape=label_arr.shape,
                method=partitioning_method,
                kernel_size=list(kernel_size.values()),
                stride=list(stride.values()),
                buffer=list(buffer.values()),
                include_last=include_last,
                ensure_slice_size=ensure_slice_size,
            )
            if n_partitions_per_label is not None:
                n_to_select = min(len(partitions_list_slices), n_partitions_per_label)
                partitions_list_slices = partitions_list_slices[0:n_to_select]
        else:
            partitions_list_slices = [label_bbox_slices]

        # --------------------------------------------------------------------.
        # Retrieve patches list_slices from partitions list slices
        patches_list_slices = _get_patches_from_partitions_list_slices(
            partitions_list_slices=partitions_list_slices,
            label_arr=label_arr,
            variable_arr=variable_arr,
            label_id=label_id,
            patch_size=list(patch_size.values()),
            centered_on=centered_on,
            n_patches_per_partition=n_patches_per_partition,
            padding=list(padding.values()),
            verbose=verbose,
        )

        # ---------------------------------------------------------------------.
        # Retrieve patches isel_dictionaries
        partitions_isel_dicts = _get_list_isel_dicts(partitions_list_slices, dims=dims)
        patches_isel_dicts = _get_list_isel_dicts(patches_list_slices, dims=dims)

        n_to_select = min(len(patches_isel_dicts), n_patches_per_label)
        patches_isel_dicts = patches_isel_dicts[0:n_to_select]

        # --------------------------------------------------------------------.
        # If debug=True, plot patches boundaries
        if debug and label_arr.ndim == 2:
            _ = plot_label_patch_extraction_areas(
                xr_obj,
                label_name=label_name,
                patches_isel_dicts=patches_isel_dicts,
                partitions_isel_dicts=partitions_isel_dicts,
            )
            plt.show()

        # ---------------------------------------------------------------------.
        # Return isel_dicts
        if grouped_by_labels_id:
            patch_counter += 1
            if patch_counter > n_patches:
                break_flag = True
            else:
                yield label_id, patches_isel_dicts
        else:
            for isel_dict in patches_isel_dicts:
                patch_counter += 1
                if patch_counter > n_patches:
                    break_flag = True
                else:
                    yield label_id, isel_dict
        if break_flag:
            break
    # ---------------------------------------------------------------------.


def get_patches_isel_dict_from_labels(
    xr_obj,
    label_name,
    patch_size,
    variable=None,
    # Output options
    n_patches=np.Inf,
    n_labels=None,
    labels_id=None,
    # Label Patch Extraction Settings
    centered_on="max",
    padding=0,
    n_patches_per_label=np.Inf,
    n_patches_per_partition=1,
    # Label Tiling/Sliding Options
    partitioning_method=None,
    n_partitions_per_label=None,
    kernel_size=None,
    buffer=0,
    stride=None,
    include_last=True,
    ensure_slice_size=True,
    debug=False,
    verbose=False,
):
    gen = _get_patches_isel_dict_generator(
        xr_obj=xr_obj,
        label_name=label_name,
        patch_size=patch_size,
        variable=variable,
        n_patches=n_patches,
        n_labels=n_labels,
        labels_id=labels_id,
        grouped_by_labels_id=True,
        # Patch extraction options
        centered_on=centered_on,
        padding=padding,
        n_patches_per_label=n_patches_per_label,
        n_patches_per_partition=n_patches_per_partition,
        # Tiling/Sliding settings
        partitioning_method=partitioning_method,
        n_partitions_per_label=n_partitions_per_label,
        kernel_size=kernel_size,
        buffer=buffer,
        stride=stride,
        include_last=include_last,
        ensure_slice_size=ensure_slice_size,
        debug=debug,
        verbose=verbose,
    )
    dict_isel_dicts = {int(label_id): list_isel_dicts for label_id, list_isel_dicts in gen}
    return dict_isel_dicts


def get_patches_from_labels(
    xr_obj,
    label_name,
    patch_size,
    variable=None,
    # Output options
    n_patches=np.Inf,
    n_labels=None,
    labels_id=None,
    highlight_label_id=True,
    # Label Patch Extraction Options
    centered_on="max",
    padding=0,
    n_patches_per_label=np.Inf,
    n_patches_per_partition=1,
    # Label Tiling/Sliding Options
    partitioning_method=None,
    n_partitions_per_label=None,
    kernel_size=None,
    buffer=0,
    stride=None,
    include_last=True,
    ensure_slice_size=True,
    debug=False,
    verbose=False,
):
    """
    Routines to extract patches around labels.

    Create a generator extracting (from a prelabeled xr.Dataset) a patch around:

    - a label point
    - a label bounding box

    If 'centered_on' is specified, output patches are guaranteed to have equal shape !
    If 'centered_on' is not specified, output patches are guaranteed to have only have a minimum shape !

    If you want to extract the patch around the label bounding box, 'centered_on'
    must not be specified.

    If you want to extract the patch around a label point, the 'centered_on'
    method must be specified. If the identified point is close to an array boundary,
    the patch is expanded toward the valid directions.

    Tiling or sliding enables to split/slide over each label and extract multiple patch
    for each tile.

    tiling=True
    - centered_on = "centroid" (tiling around labels bbox)
    - centered_on = "center_of_mass" (better coverage around label)

    sliding=True
    - centered_on = "center_of_mass" (better coverage around label) (further data coverage)

    Only one parameter between n_patches and labels_id can be specified.

    Parameters
    ----------
    xr_obj : xr.Dataset
        xr.Dataset with a label array named label_name.
    label_name : str
        Name of the variable/coordinate representing the label array.
    patch_size : (int, tuple)
        The dimensions of the n-dimensional patch to extract.
        Only positive values (>1) are allowed.
        The value -1 can be used to specify the full array dimension shape.
        If the centered_on method is not 'label_bbox', all output patches
        are ensured to have the same shape.
        Otherwise, if 'centered_on'='label_bbox', the patch_size argument defines
        defined the minimum n-dimensional shape of the output patches.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    n_patches : int, optional
        Maximum number of patches to extract.
        The default (np.Inf) enable to extract all available patches allowed by the
        specified patch extraction criteria.
    labels_id : list, optional
        List of labels for which to extract the patch.
        If None, it extracts the patches by label order (1, 2, 3, ...)
        The default is None.
    n_labels : int, optional
        The number of labels for which extract patches.
        If None (the default), it extract patches for all labels.
        This argument can be specified only if labels_id is unspecified !
    highlight_label_id : (bool), optional
        If True, the label_name array of each patch is modified to contain only
        the label_id used to select the patch.
    variable : str, optional
        Dataset variable to use to identify the patch center when centered_on is defined.
        This is required only for centered_on='max', 'min' or the custom function.

    centered_on : (str, callable), optional
        The centered_on method characterize the point around which the patch is extracted.
        Valid pre-implemented centered_on methods are 'label_bbox', 'max', 'min',
        'centroid', 'center_of_mass', 'random'.
        The default method is 'max'.

        If 'label_bbox' it extract the patches around the (padded) bounding box of the label.
        If 'label_bbox',the output patch sizes are only ensured to have a minimum patch_size,
        and will likely be of different size.
        Otherwise, the other methods guarantee that the output patches have a common shape.

        If centered_on is 'max', 'min' or a custom function, the 'variable' must be specified.
        If centered_on is a custom function, it must:
            - return None if all array values are non-finite (i.e np.nan)
            - return a tuple with same length as the array shape.
    padding : (int, tuple, dict), optional
        The padding to apply in each direction around a label prior to
        partitioning (tiling/sliding) or direct patch extraction.
        The default, 0, applies 0 padding in every dimension.
        Negative padding values are allowed !
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    n_patches_per_label: int, optional
        The maximum number of patches to extract for each label.
        The default (np.Inf) enables to extract all the available patches per label.
        n_patches_per_label must be larger than n_patches_per_partition !
    n_patches_per_partition, int, optional
        The maximum number of patches to extract from each label partition.
        The default values is 1.
        This method can be specified only if centered_on='random' or a callable.
    partitioning_method : str
        Whether to retrieve 'tiling' or 'sliding' slices.
        If 'tiling', partition start slices are separated by stride + kernel_size
        If 'sliding', partition start slices are separated by stride.
    n_partitions_per_label : int, optional
        The maximum number of partitions to extract for each label.
        The default (None) enables to extract all the available partitions per label.
    stride : (int, tuple, dict), optional
        If partitioning_method is 'sliding'', default stride is set to 1.
        If partitioning_method is 'tiling', default stride is set to 0.
        Step size between slices.
        When 'tiling', a positive stride make partition slices to not overlap and not touch,
        while a negative stride make partition slices to overlap by 'stride' amount.
        If stride is 0, the partition slices are contiguous (no spacing between partitions).
        When 'sliding', only a positive stride (>= 1) is allowed.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    kernel_size: (int, tuple, dict), optional
        The shape of the desired partitions.
        Only positive values (>1) are allowed.
        The value -1 can be used to specify the full array dimension shape.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    buffer: (int, tuple, dict), optional
        The default is 0.
        Value by which to enlarge a partition on each side.
        The final partition size should be kernel_size + buffer.
        If 'tiling' and stride=0, a positive buffer value corresponds to
        the amount of overlap between each partition.
        Depending on min_start and max_stop values, buffering might cause
        border partitions to not have same sizes.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    include_last : bool, optional
        Whether to include the last partition if it does not match the kernel_size.
        The default is True.
    ensure_slice_size : False, optional
        Used only if include_last is True.
        If False, the last partition will not have the specified kernel_size.
        If True,  the last partition is enlarged to the specified kernel_size by
        tentatively expandinf it on both sides (accounting for min_start and max_stop).

    Yields
    ------
    (xr.Dataset or xr.DataArray)
        A xarray object patch.

    """
    # Define patches isel dictionary generator
    patches_isel_dicts_gen = _get_patches_isel_dict_generator(
        xr_obj=xr_obj,
        label_name=label_name,
        patch_size=patch_size,
        variable=variable,
        n_patches=n_patches,
        n_labels=n_labels,
        labels_id=labels_id,
        grouped_by_labels_id=False,
        # Label Patch Extraction Options
        centered_on=centered_on,
        padding=padding,
        n_patches_per_label=n_patches_per_label,
        n_patches_per_partition=n_patches_per_partition,
        # Tiling/Sliding Options
        partitioning_method=partitioning_method,
        n_partitions_per_label=n_partitions_per_label,
        kernel_size=kernel_size,
        buffer=buffer,
        stride=stride,
        include_last=include_last,
        ensure_slice_size=ensure_slice_size,
        debug=debug,
        verbose=verbose,
    )

    # Extract the patches
    for label_id, isel_dict in patches_isel_dicts_gen:
        xr_obj_patch = _extract_xr_patch(
            xr_obj=xr_obj,
            label_name=label_name,
            isel_dict=isel_dict,
            label_id=label_id,
            highlight_label_id=highlight_label_id,
        )

        # Return the patch around the label
        yield label_id, xr_obj_patch
