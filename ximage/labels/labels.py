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
"""Labels identification."""

# import dask_image.ndmeasure
# from dask_image.ndmeasure import as dask_label_image
import dask.array
import dask_image.ndmeasure
import numpy as np
import xarray as xr
from skimage.measure import label as label_image
from skimage.morphology import dilation as skimage_dilation
from skimage.morphology import disk

from ximage.utils.checks import are_all_natural_numbers

# TODO:
# - Enable to label in n-dimensions
#   - (2D+VERTICAL) --> CORE PROFILES
#   - (2D+TIME) --> TRACKING


####--------------------------------------------------------------------------.


def _binary_dilation(mask, footprint):
    mask = skimage_dilation(mask, footprint=footprint)
    return mask  # noqa: RET504


def _mask_buffer(mask, footprint):
    """Dilate the mask by n pixel in all directions.

    If footprint = 0 or None, no dilation occur.
    If footprint is a positive integer, it create a disk(footprint)
    If footprint is a 2D array, it must represent the neighborhood expressed
    as a 2-D array of 1's and 0's.
    For more info: https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation

    """
    # scikitimage > 0.19
    if not isinstance(footprint, (int, np.ndarray, type(None))):
        raise TypeError("`footprint` must be an integer, numpy 2D array or None.")
    if isinstance(footprint, np.ndarray) and footprint.ndim != 2:
        raise ValueError("If providing the footprint for dilation as np.array, it must be 2D.")
    if isinstance(footprint, int):
        if footprint < 0:
            raise ValueError("Footprint must be equal or larger than 1.")
        footprint = None if footprint == 0 else disk(radius=footprint)
    # Apply dilation
    if footprint is not None:
        mask = _binary_dilation(mask, footprint=footprint)
    return mask


def _check_array(arr):
    """Check array and return a numpy.ndarray."""
    shape = arr.shape
    if len(shape) != 2:
        raise ValueError("Expecting a 2D array.")
    if np.any(np.array(shape) == 0):
        raise ValueError("Expecting non-zero dimensions.")

    # Convert to numpy array
    return np.asanyarray(arr)


def _no_labels_result(arr, return_labels_stats):
    """Define results for array without labels."""
    labels = np.zeros(arr.shape)
    n_labels = 0
    values = []
    if return_labels_stats:
        return labels, n_labels, values
    return labels


def _check_sort_by(sort_by):
    """Check ``sort_by`` argument."""
    if not (callable(sort_by) or isinstance(sort_by, str)):
        raise TypeError("'sort_by' must be a string or a function.")
    if isinstance(sort_by, str):
        valid_stats = [
            "area",
            "maximum",
            "minimum",
            "mean",
            "median",
            "sum",
            "standard_deviation",
            "variance",
        ]
        if sort_by not in valid_stats:
            raise ValueError(f"Valid 'sort_by' values are: {valid_stats}.")


def _check_stats(stats):
    """Check ``stats`` argument."""
    if not (callable(stats) or isinstance(stats, str)):
        raise TypeError("'stats' must be a string or a function.")
    if isinstance(stats, str):
        valid_stats = [
            "area",
            "maximum",
            "minimum",
            "mean",
            "median",
            "sum",
            "standard_deviation",
            "variance",
        ]
        if stats not in valid_stats:
            raise ValueError(f"Valid 'stats' values are: {valid_stats}.")
    # TODO: check stats function works on a dummy array (reduce to single value)
    return stats


def _get_label_value_stats(arr, label_arr, label_indices=None, stats="area", labeled_comprehension_kwargs=None):
    """Compute label value statistics over which to later sort on.

    If ``label_indices`` is None, by default would return the stats of the entire array.
    If ``label_indices`` is 0, return ``np.nan``.
    If ``label_indices`` is not inside ``label_arr``, return 0.
    """
    # Check stats argument and label indices
    if labeled_comprehension_kwargs is None:
        labeled_comprehension_kwargs = {}
    stats = _check_stats(stats)
    if label_indices is None:
        label_indices = np.unique(label_arr)
    # Compute labels stats values
    if callable(stats):
        labeled_comprehension_kwargs.setdefault("out_dtype", float)
        labeled_comprehension_kwargs.setdefault("default", None)
        labeled_comprehension_kwargs.setdefault("pass_positions", False)
        values = dask_image.ndmeasure.labeled_comprehension(
            image=arr,
            label_image=label_arr,
            index=label_indices,
            func=stats,
            **labeled_comprehension_kwargs,
        )
    else:
        func = getattr(dask_image.ndmeasure, stats)
        values = func(image=arr, label_image=label_arr, index=label_indices)
    # Compute values
    return values.compute()


def _get_labels_stats(
    arr,
    label_arr,
    label_indices=None,
    stats="area",
    sort_decreasing=True,
    labeled_comprehension_kwargs=None,
):
    """Return label and label statistics sorted by statistic value."""
    if labeled_comprehension_kwargs is None:
        labeled_comprehension_kwargs = {}
    if label_indices is None:
        label_indices = np.unique(label_arr)

    # Get labels area values
    values = _get_label_value_stats(
        arr,
        label_arr=label_arr,
        label_indices=label_indices,
        stats=stats,
        labeled_comprehension_kwargs=labeled_comprehension_kwargs,
    )
    # Get sorting index based on values
    sort_index = np.argsort(values)[::-1] if sort_decreasing else np.argsort(values)

    # Sort values
    values = values[sort_index]
    label_indices = label_indices[sort_index]

    return label_indices, values


def _vec_translate(arr, my_dict):
    """Remap array <value> based on the dictionary key-value pairs.

    This function is used to redefine label array integer values based on the
    label area_size/max_intensity value.

    """
    # TODO:  Remove keys not in arr to speed up maybe
    return np.vectorize(my_dict.__getitem__)(arr)


def _get_labels_with_requested_occurrence(label_arr, vmin, vmax):
    """Get label indices with requested occurrence."""
    # Compute label occurrence
    label_indices, label_occurrence = np.unique(label_arr, return_counts=True)

    # Remove label 0 and associate pixel count if present
    if label_indices[0] == 0:
        label_indices = label_indices[1:]
        label_occurrence = label_occurrence[1:]
    # Get index with required occurrence
    valid_area_indices = np.where(np.logical_and(label_occurrence >= vmin, label_occurrence <= vmax))[0]
    # Return list of valid label indices
    return label_indices[valid_area_indices] if len(valid_area_indices) > 0 else []


def _ensure_valid_label_arr(label_arr):
    """Ensure ``label_arr`` does contain only positive values.

    NaN values are converted to 0.
    The output array type is int.
    """
    # Ensure data are numpy
    label_arr = np.asanyarray(label_arr)

    # Set NaN to 0
    label_arr[np.isnan(label_arr)] = 0

    # Check that label arr values are positive integers
    if not are_all_natural_numbers(label_arr.flatten(), zero_allowed=True):
        raise ValueError("The label array must contain only positive integers.")

    # Ensure label array is integer dtype
    return label_arr.astype(int)


def _ensure_valid_label_indices(label_indices):
    """Ensure valid label indices are integers and does not contains 0 and NaN."""
    label_indices = np.delete(label_indices, np.where(label_indices == 0)[0].flatten())
    label_indices = np.delete(label_indices, np.where(np.isnan(label_indices))[0].flatten())
    return label_indices.astype(int)


def get_label_indices(arr):
    """Get label indices from numpy.ndarray, dask.Array and xarray.DataArray.

    It removes 0 and ``np.NaN`` values. Output type is ``int``.
    """
    arr = np.asanyarray(arr)
    arr = arr[~np.isnan(arr)]
    arr = arr.astype(int)  # otherwise precision error in unique
    label_indices = np.unique(arr)
    return _ensure_valid_label_indices(label_indices)


def _check_unique_label_indices(label_indices):
    _, c = np.unique(label_indices, return_counts=True)
    if np.any(c > 1):
        raise ValueError("'label_indices' must be uniques.")


def _get_new_label_value_dict(label_indices, max_label):
    """Create dictionary mapping from current label value to new label value."""
    # Initialize dictionary with keys corresponding to all possible labels indices
    val_dict = dict.fromkeys(range(0, max_label + 1), 0)

    # Update the dictionary keys with the selected label_indices
    # - Assume 0 not in label_indices
    n_labels = len(label_indices)
    label_indices = label_indices.tolist()
    label_indices_new = np.arange(1, n_labels + 1, dtype=int).tolist()
    val_dict.update(dict(zip(label_indices, label_indices_new, strict=True)))
    return val_dict


def _np_redefine_label_array(label_arr, label_indices=None):
    """Relabel a numpy/dask array from 0 to len(label_indices)."""
    # Ensure data are numpy
    label_arr = np.asanyarray(label_arr)

    if label_indices is None:
        label_indices = np.unique(label_arr)
    else:
        _check_unique_label_indices(label_indices)

    # Ensure label indices are integer, without 0 and NaN
    label_indices = _ensure_valid_label_indices(label_indices)

    # Ensure label array values are integer
    label_arr = _ensure_valid_label_arr(label_arr)  # output is int, without NaN

    # Check there are label_indices
    if len(label_indices) == 0:
        raise ValueError("No labels available.")

    # Compute max label index
    max_label = max(label_indices)

    # Set to 0 labels in label_arr larger than max_label
    # - These are some of the labels that were set to 0 because of mask or area filtering
    label_arr[label_arr > max_label] = 0

    # Initialize dictionary with keys corresponding to all possible labels indices
    val_dict = _get_new_label_value_dict(label_indices, max_label)

    # Redefine the id of the labels
    return _vec_translate(label_arr, val_dict)


def _xr_redefine_label_array(dataarray, label_indices=None):
    """Relabel a xarray.DataArray from 0 to len(label_indices)."""
    relabeled_arr = _np_redefine_label_array(dataarray.data, label_indices=label_indices)
    da_label = dataarray.copy()
    da_label.data = relabeled_arr
    return da_label


def redefine_label_array(data, label_indices=None):
    """Redefine labels of a label array from 0 to len(label_indices).

    If ``label_indices`` is ``None``, it takes the unique values of ``label_arr``.
    If ``label_indices`` contains a 0, it is discarded !
    If ``label_indices`` is not unique, raise an error !

    Native label values not present in label_indices are set to 0.
    The first label in ``label_indices`` becomes 1, the second 2, and so on.
    """
    if isinstance(data, xr.DataArray):
        return _xr_redefine_label_array(data, label_indices=label_indices)
    if isinstance(data, (np.ndarray, dask.array.Array)):
        return _np_redefine_label_array(data, label_indices=label_indices)
    raise TypeError(f"This method does not accept {type(data)}")


def get_data_array(xr_obj, variable=None):
    """Check xarray object and variable validity."""
    # Check inputs
    if not isinstance(xr_obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("'xr_obj' must be a xr.Dataset or xr.DataArray.")
    if isinstance(xr_obj, xr.Dataset):
        # Check valid variable is specified
        if variable is None:
            raise ValueError("An xr.Dataset 'variable' must be specified.")
        if variable not in xr_obj.data_vars:
            raise ValueError(f"'{variable}' is not a variable of the xr.Dataset.")
    elif variable is not None:
        raise ValueError("'variable' must not be specified when providing a xr.DataArray.")
    # Return DataArray
    return xr_obj[variable] if isinstance(xr_obj, xr.Dataset) else xr_obj


def check_core_dims(core_dims, data_array):
    """Check core_dims argument and infer if needed."""
    # Infer core_dims if 2D array
    if data_array.ndim == 2:
        core_dims = tuple(data_array.dims) if core_dims is None else tuple(core_dims)
    # Otherwise should be specified
    else:
        if core_dims is None:
            raise ValueError(
                "For DataArray with ndim > 2, `core_dims` must be specified.",
            )
        core_dims = tuple(core_dims)

    # Check core_dims are two  (currently) !
    if len(core_dims) != 2:
        raise ValueError("`core_dims` must contain exactly two dimensions. 3D-array labelling not yet implemented.")

    # Check valid core_dims
    missing = set(core_dims) - set(data_array.dims)
    if missing:
        raise ValueError(
            f"`core_dims` {core_dims} are not all dimensions of the DataArray. " f"Missing: {missing}",
        )
    return core_dims


def _get_labels(
    arr,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
    labeled_comprehension_kwargs=None,
    return_labels_stats=True,
):
    """
    Function deriving the labels array and associated labels info.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be labelled.
    min_value_threshold : float, optional
        The minimum value to define the interior of a label.
        The default is -np.inf.
    max_value_threshold : float, optional
        The maximum value to define the interior of a label.
        The default is np.inf.
    min_area_threshold : float, optional
        The minimum number of connected pixels to be defined as a label.
        The default is 1.
    max_area_threshold : float, optional
        The maximum number of connected pixels to be defined as a label.
        The default is np.inf.
    footprint : (int, numpy.ndarray or None), optional
        This argument enables to dilate the mask derived after applying
        min_value_threshold and max_value_threshold.
        If footprint = 0 or None, no dilation occur.
        If footprint is a positive integer, it create a disk(footprint)
        If footprint is a 2D array, it must represent the neighborhood expressed
        as a 2-D array of 1's and 0's.
        The default is None (no dilation).
    sort_by : (callable or str), optional
        A function or statistics to define the order of the labels.
        Valid string statistics are "area", "maximum", "minimum", "mean",
        "median", "sum", "standard_deviation", "variance".
        The default is "area".
    sort_decreasing : bool, optional
        If True, sort labels by decreasing 'sort_by' value.
        The default is True.
    labeled_comprehension_kwargs : dict, optional
        Additional arguments to be passed to dask_image.ndmeasure.labeled_comprehension
        if sort_by is a callable. May contain
            out_dtype : dtype, optional
                Dtype to use for result.
                The default is float.
            default : (int, float or None), optional
                Default return value when a element of index does not exist in the label array.
                The default is None.
            pass_positions : bool, optional
                If True, pass linear indices to 'sort_by' as a second argument.
                The default is False.
        The default is {}.
    return_labels_stats: bool
        Whether to return label statistics. The default is True.
        If False, it returns just the labelled array.

    Returns
    -------
    labels_arr, numpy.ndarray
        Label array. 0 values corresponds to no label.
    n_labels, int
        Number of labels in the labels array.
    values, numpy.arrays
        Array of length n_labels with the stats values associated to each label.
    """
    # ---------------------------------.
    # TODO: this could be extended to work with dask >2D array
    # - dask_image.ndmeasure.label  https://image.dask.org/en/latest/dask_image.ndmeasure.html
    # - dask_image.ndmorph.binary_dilation https://image.dask.org/en/latest/dask_image.ndmorph.html#dask_image.ndmorph.binary_dilation

    # ---------------------------------.
    # Check array validity
    if labeled_comprehension_kwargs is None:
        labeled_comprehension_kwargs = {}
    arr = _check_array(arr)

    # ---------------------------------.
    # Define masks
    # - mask_native: True when between min and max thresholds
    # - mask_nan: True where is not finite (inf or nan)
    mask_native = np.logical_and(arr >= min_value_threshold, arr <= max_value_threshold)
    mask_nan = ~np.isfinite(arr)
    # ---------------------------------.
    # Dilate (buffer) the native mask
    # - This enable to assign closely connected mask_native areas to the same label
    mask = _mask_buffer(mask_native, footprint=footprint)

    # ---------------------------------.
    # Get area labels
    # - 0 represent the outer area
    label_arr = label_image(mask)  # 0.977-1.37 ms

    # mask = mask.astype(int)
    # labels, num_features = dask_label_image(mask) # THIS WORK in n-dimensions
    # %time labels = labels.compute()    # 5-6.5 ms

    # ---------------------------------.
    # Count initial label occurrence
    label_indices = np.unique(label_arr, return_counts=False)
    n_initial_labels = len(label_indices)
    if n_initial_labels == 1:  # only 0 label
        return _no_labels_result(arr, return_labels_stats=return_labels_stats)

    # ---------------------------------.
    # Set areas outside the mask_native to label value 0
    label_arr[~mask_native] = 0

    # Set NaN pixels to label value 0
    label_arr[mask_nan] = 0

    # ---------------------------------.
    # Filter label by area
    label_indices = _get_labels_with_requested_occurrence(
        label_arr=label_arr,
        vmin=min_area_threshold,
        vmax=max_area_threshold,
    )
    if len(label_indices) == 0:
        return _no_labels_result(arr, return_labels_stats=return_labels_stats)

    # ---------------------------------.
    # Sort labels by statistics (i.e. label area, label max value ...)
    label_indices, values = _get_labels_stats(
        arr=arr,
        label_arr=label_arr,
        label_indices=label_indices,
        stats=sort_by,
        sort_decreasing=sort_decreasing,
        labeled_comprehension_kwargs=labeled_comprehension_kwargs,
    )
    # ---------------------------------.
    # TODO: optionally here calculate a list of label_stats
    # --> values would be a n_label_stats x n_labels array !
    # --> dask_image.ndmeasure.labeled_comprehension

    # ---------------------------------.
    # Relabel labels array (from 1 to n_labels)
    labels_arr = redefine_label_array(label_arr, label_indices=label_indices)
    n_labels = len(label_indices)

    # ---------------------------------.
    # Return results
    if return_labels_stats:
        return labels_arr, n_labels, values
    return labels_arr


def label(
    xr_obj,
    *,
    variable=None,
    core_dims=None,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
    labeled_comprehension_kwargs=None,
    label_name="label",
):
    """
    Compute labels and and add as a coordinates to an xarray object.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    variable : str, optional
        Dataset variable to exploit to derive the labels array.
        Must be specified only if the input object is an `xarray.Dataset`.
    core_dims : tuple of str, optional
        Names of the two dimensions along which the labeling is applied.
        If the xarray DataArray is two-dimensional and ``core_dims`` is not provided,
        the core dimensions are inferred automatically from DataArray.dims.
        If the xarray DataArray  has more than two dimensions, ``core_dims`` must be
        specified explicitly. In this case, labeling is applied independently
        over all remaining (non-core) dimensions.
        Example: for a 3D DataArray with dimensions ``(x, y, time)``,
        use ``core_dims=("x", "y")`` to apply labeling to each timestep.
    min_value_threshold : float, optional
        The minimum value to define the interior of a label.
        The default is ``-np.inf``.
    max_value_threshold : float, optional
        The maximum value to define the interior of a label.
        The default is ``np.inf``.
    min_area_threshold : float, optional
        The minimum number of connected pixels to be defined as a label.
        The default is 1.
    max_area_threshold : float, optional
        The maximum number of connected pixels to be defined as a label.
        The default is ``np.inf``.
    footprint : int, numpy.ndarray or None, optional
        This argument enables to dilate the mask derived after applying
        min_value_threshold and max_value_threshold.
        If ``footprint = 0`` or ``None``, no dilation occur.
        If ``footprint`` is a positive integer, it create a ``disk(footprint)``
        If ``footprint`` is a 2D array, it must represent the neighborhood expressed
        as a 2-D array of 1's and 0's.
        The default is ``None`` (no dilation).
    sort_by : callable or str, optional
        A function or statistics to define the order of the labels.
        Valid string statistics are ``"area"``, ``"maximum"``, ``"minimum"``, ``"mean"``,
        ``"median"``, ``"sum"``, ``"standard_deviation"``, ``"variance"``.
        The default is ``"area"``.
    sort_decreasing : bool, optional
        If ``True``, sort labels by decreasing ``sort_by`` value.
        The default is ``True``.
    labeled_comprehension_kwargs : dict, optional
        Additional arguments to be passed to `dask_image.ndmeasure.labeled_comprehension`.
        if ``sort_by`` is a callable.

    Returns
    -------
    xr_obj : (xarray.DataArray or xarray.Dataset)
        xarray object with the new label coordinate.
        In the label coordinate, non-labels values are set to np.nan.
    """
    # Check xarray input
    if labeled_comprehension_kwargs is None:
        labeled_comprehension_kwargs = {}

    # Retrieve datarray to label
    data_array = get_data_array(xr_obj=xr_obj, variable=variable)

    # Check arguments
    _check_sort_by(sort_by)
    core_dims = check_core_dims(core_dims, data_array)

    # Define kwargs
    kwargs = {
        "min_value_threshold": min_value_threshold,
        "max_value_threshold": max_value_threshold,
        "min_area_threshold": min_area_threshold,
        "max_area_threshold": max_area_threshold,
        "footprint": footprint,
        "sort_by": sort_by,
        "sort_decreasing": sort_decreasing,
        "labeled_comprehension_kwargs": labeled_comprehension_kwargs,
        "return_labels_stats": False,
    }

    # Apply over non-core dimensions
    da_labels = xr.apply_ufunc(
        _get_labels,
        data_array,
        kwargs=kwargs,
        input_core_dims=[list(core_dims)],
        output_core_dims=[list(core_dims)],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"parameters": 3}},
    )

    # If input array was in memory compute labels
    if hasattr(data_array, "chunks"):
        da_labels = da_labels.compute()
        if da_labels.max() == 0:
            raise ValueError("No labels identified. You might want to change the labeling parameters.")

    # Conversion to DataArray if needed
    da_labels.name = f"labels_{sort_by}"
    da_labels.attrs = {}

    # Set labels values == 0 to np.nan (useful for plotting)
    da_labels = da_labels.where(da_labels > 0)

    # Assign label to xr.DataArray  coordinate
    return xr_obj.assign_coords({label_name: da_labels})


def highlight_label(xr_obj, label_name, label_id):
    """Set all labels values to 0 except for 'label_id'."""
    xr_obj = xr_obj.copy(deep=True)  # required otherwise overwrite original data
    label_arr = xr_obj[label_name].data
    label_arr[label_arr != label_id] = 0
    xr_obj[label_name].data = label_arr
    return xr_obj
