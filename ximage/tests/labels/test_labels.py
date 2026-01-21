import dask.array
import numpy as np
import pytest
import xarray as xr

from ximage.labels import labels
from ximage.labels.labels import (
    check_core_dims,
)


def apply_to_all_array_types(func, array, *args, **kwargs):
    """Apply a function to numpy.ndarray, dask.Array, and xarray.DataArray."""
    np_array = np.array(array)
    dask_array = dask.array.from_array(array)
    xr_array = xr.DataArray(array)

    for x_array in [np_array, dask_array, xr_array]:
        func(x_array, *args, **kwargs)


# ############################################################################
# Tests for public functions
# ############################################################################


def test_get_label_indices():
    """Extract label indices from array with NaN and duplicate values."""
    array = [float("nan"), 0, 1, 1, 2.001, 4, 1]
    expected_indices = np.array([1, 2, 4])

    def check(test_array):
        label_indices_returned = labels.get_label_indices(test_array)
        np.testing.assert_array_equal(label_indices_returned, expected_indices)
        assert label_indices_returned.dtype in (np.int64, np.int32)

    apply_to_all_array_types(check, array)


class TestRedefineLabelArray:
    """Unit tests for redefine_label_array function."""

    def test_redefine_labels_with_provided_indices(self):
        """Remap label values using explicitly provided label indices."""
        array = [3, 3, 4, 6]
        label_indices = np.array([3, 4])
        expected = np.array([1, 1, 2, 0])

        def check(test_array):
            result = labels.redefine_label_array(test_array, label_indices)
            np.testing.assert_array_equal(result, expected)

        apply_to_all_array_types(check, array)

    def test_redefine_labels_with_default_indices(self):
        """Remap labels to sequential integers starting from 1."""
        array = [3, 3, 4, 6]
        expected = np.array([1, 1, 2, 3])

        def check(test_array):
            result = labels.redefine_label_array(test_array)
            np.testing.assert_array_equal(result, expected)

        apply_to_all_array_types(check, array)

    def test_ignore_zero_in_label_indices(self):
        """Skip zero label indices during remapping."""
        array = [3, 3, 4, 6]
        label_indices_with_0 = np.array([0, 3, 4])
        expected = np.array([1, 1, 2, 0])

        def check(test_array):
            result = labels.redefine_label_array(test_array, label_indices_with_0)
            np.testing.assert_array_equal(result, expected)

        apply_to_all_array_types(check, array)

    def test_raise_error_for_duplicate_label_indices(self):
        """Raise ValueError when label indices contain duplicates."""
        array = [3, 3, 4, 6]
        label_indices_with_duplicate = np.array([3, 3, 4])

        def check(test_array):
            with pytest.raises(ValueError):
                labels.redefine_label_array(test_array, label_indices_with_duplicate)

        apply_to_all_array_types(check, array)

    def test_raise_error_for_invalid_array_type(self):
        """Raise TypeError when array is not xr.DataArray or np.ndarray."""
        array = [3, 3, 4, 6]
        with pytest.raises(TypeError):
            labels.redefine_label_array(array)


class TestLabel:
    """Unit tests for label function."""

    def test_raise_error_for_non_dataarray_input(self):
        """Raise TypeError when input is not xr.DataArray or xr.Dataset."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        with pytest.raises(TypeError):
            labels.label(array)

    def test_label_dataarray_with_all_parameters(self):
        """Apply all filtering and sorting parameters correctly to DataArray."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        data_array = xr.DataArray(array)

        footprint = np.array([[1, 1], [1, 1]])
        data_array_returned = labels.label(
            data_array,
            min_value_threshold=3,
            max_value_threshold=8,
            min_area_threshold=2,
            max_area_threshold=4,
            footprint=footprint,
            sort_by="mean",
            sort_decreasing=False,
        )

        expected = np.array(
            [
                [_, _, _, _, 1],
                [_, _, _, _, 1],
                [_, _, _, _, _],
                [2, _, _, _, _],
                [2, 2, _, 2, _],
            ],
        )
        np.testing.assert_array_equal(
            data_array_returned["label"].to_numpy(),
            expected,
        )

    def test_raise_error_when_no_labels_found(self):
        """Raise ValueError when value filtering results in no labels."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        data_array = xr.DataArray(array)

        with pytest.raises(ValueError):
            labels.label(data_array, max_value_threshold=0)

    def test_label_dataset_with_variable_name(self):
        """Apply labeling to a specific variable in a Dataset."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        variable_name = "data"
        dataset = xr.Dataset({variable_name: xr.DataArray(array)})
        dataset_returned = labels.label(dataset, variable=variable_name)

        expected = np.array(
            [
                [3, 3, _, _, 2],
                [_, _, _, _, 2],
                [_, _, _, _, _],
                [1, _, _, _, _],
                [1, 1, _, 4, _],
            ],
        )
        np.testing.assert_array_equal(
            dataset_returned.coords["label"].values,
            expected,
        )


class TestLabelWithCustomSorting:
    """Unit tests for label function with custom sorting."""

    def test_label_with_custom_sort_function(self):
        """Apply custom sorting function to determine label order."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )

        def my_sort_function(array):
            return np.mean(array)

        data_array = xr.DataArray(array)
        footprint = np.array([[1, 1], [1, 1]])
        labels_array = labels.label(
            data_array,
            min_value_threshold=3,
            max_value_threshold=8,
            min_area_threshold=2,
            max_area_threshold=4,
            footprint=footprint,
            sort_by=my_sort_function,
            sort_decreasing=False,
            labeled_comprehension_kwargs={"out_dtype": np.float16},
        )

        expected = np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 2, 0, 2, 0],
            ],
            dtype="float",
        )
        expected[expected == 0] = np.nan

        assert isinstance(labels_array, xr.DataArray)
        np.testing.assert_allclose(labels_array["label"].data, expected, equal_nan=True)


class TestHighlightLabel:
    """Unit tests for highlight_label function."""

    def test_highlight_single_label_in_dataset(self):
        """Extract and highlight a specific label from a labeled dataset."""
        _ = float("nan")
        labels_array = xr.DataArray(
            [
                [_, 1, 2],
                [3, _, 2],
                [3, 3, _],
            ],
        )
        label_name = "test_label"
        dataset = xr.Dataset({label_name: labels_array})
        dataset_returned = labels.highlight_label(dataset, label_name, 2)

        expected = np.array(
            [
                [0, 0, 2],
                [0, 0, 2],
                [0, 0, 0],
            ],
        )
        np.testing.assert_array_equal(dataset_returned[label_name].values, expected)


# ############################################################################
# Tests for internal functions
# ############################################################################


class TestMaskBuffer:
    """Unit tests for _mask_buffer function."""

    def test_no_dilation_with_none_buffer(self):
        """Return original mask when buffer is None."""
        mask_ini = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        result = labels._mask_buffer(mask_ini, None)
        np.testing.assert_array_equal(result, mask_ini)

    def test_no_dilation_with_zero_buffer(self):
        """Return original mask when buffer is zero."""
        mask_ini = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        result = labels._mask_buffer(mask_ini, 0)
        np.testing.assert_array_equal(result, mask_ini)

    def test_dilation_with_integer_buffer(self):
        """Dilate mask by specified integer buffer distance."""
        mask_ini = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        expected = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
        result = labels._mask_buffer(mask_ini, 1)
        np.testing.assert_array_equal(result, expected)

    def test_dilation_with_footprint_array(self):
        """Dilate mask using custom footprint structure."""
        mask_ini = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        footprint = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
        result = labels._mask_buffer(mask_ini, footprint)
        np.testing.assert_array_equal(result, footprint)

    def test_raise_error_for_invalid_buffer_type(self):
        """Raise TypeError when buffer is not int, float, or ndarray."""
        mask_ini = np.array([[0, 1, 0]], dtype=bool)
        with pytest.raises(TypeError):
            labels._mask_buffer(mask_ini, "invalid type")

    def test_raise_error_for_negative_buffer(self):
        """Raise ValueError when buffer is negative."""
        mask_ini = np.array([[0, 1, 0]], dtype=bool)
        with pytest.raises(ValueError):
            labels._mask_buffer(mask_ini, -1)

    def test_raise_error_for_non_2d_footprint(self):
        """Raise ValueError when footprint is not 2D."""
        mask_ini = np.array([[0, 1, 0]], dtype=bool)
        non_2D_footprint = np.array([1, 1, 1])
        with pytest.raises(ValueError):
            labels._mask_buffer(mask_ini, non_2D_footprint)


class TestCheckArray:
    """Unit tests for _check_array function."""

    def test_raise_error_for_1d_array(self):
        """Raise ValueError when array is 1D instead of 2D."""
        with pytest.raises(ValueError):
            labels._check_array(np.array([1, 2, 3]))

    def test_raise_error_for_3d_array(self):
        """Raise ValueError when array is 3D instead of 2D."""
        with pytest.raises(ValueError):
            labels._check_array(np.ones((3, 3, 3)))

    def test_raise_error_for_zero_size_dimension(self):
        """Raise ValueError when array has a dimension of size 0."""
        with pytest.raises(ValueError):
            labels._check_array(np.ones((0, 3)))

    def test_convert_to_numpy_array(self):
        """Convert input array to numpy.ndarray."""
        array = np.ones((3, 3))

        def check(test_array):
            checked_array = labels._check_array(test_array)
            assert isinstance(checked_array, np.ndarray)
            np.testing.assert_array_equal(checked_array, np.asanyarray(test_array))

        apply_to_all_array_types(check, array)


class TestNoLabelsResult:
    """Unit tests for _no_labels_result function."""

    def test_return_zero_array_with_stats(self):
        """Return zero array and empty stats when no labels found."""
        array = np.ones((2, 3))
        labels_result, n_labels, values = labels._no_labels_result(array, return_labels_stats=True)

        expected = np.zeros(array.shape)
        np.testing.assert_array_equal(labels_result, expected)
        assert n_labels == 0
        assert values == []

    def test_return_zero_array_without_stats(self):
        """Return zero array without stats when return_labels_stats is False."""
        array = np.ones((2, 3))
        labels_result = labels._no_labels_result(array, return_labels_stats=False)

        expected = np.zeros(array.shape)
        np.testing.assert_array_equal(labels_result, expected)


class TestCheckSortByAndStats:
    """Unit tests for _check_sort_by and _check_stats functions."""

    def test_raise_error_for_none_sort_by(self):
        """Raise TypeError when sort_by is None."""
        with pytest.raises(TypeError):
            labels._check_sort_by(None)

    def test_raise_error_for_none_stats(self):
        """Raise TypeError when stats is None."""
        with pytest.raises(TypeError):
            labels._check_stats(None)

    def test_raise_error_for_invalid_sort_by_string(self):
        """Raise ValueError when sort_by is invalid string."""
        with pytest.raises(ValueError):
            labels._check_sort_by("invalid string")

    def test_raise_error_for_invalid_stats_string(self):
        """Raise ValueError when stats is invalid string."""
        with pytest.raises(ValueError):
            labels._check_stats("invalid string")

    def test_return_valid_stats_string(self):
        """Return valid stats string unchanged."""
        stats = "mean"
        result = labels._check_stats(stats)
        assert result == stats


class TestGetLabelValueStats:
    """Unit tests for _get_label_value_stats function."""

    def test_get_area_statistics_for_each_label(self):
        """Calculate area statistics for each label in array."""
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        label_array = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [2, 2, 0],
            ],
        )
        values = labels._get_label_value_stats(array, label_array)
        expected_area = [5, 1, 3]
        np.testing.assert_array_equal(values, expected_area)

    def test_get_custom_statistics_for_each_label(self):
        """Calculate custom statistics for each label."""
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        label_array = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [2, 2, 0],
            ],
        )

        def constant_stats(*args, **kwargs):
            return 42

        values = labels._get_label_value_stats(array, label_array, stats=constant_stats)
        expected = [42, 42, 42]
        np.testing.assert_array_equal(values, expected)


class TestGetLabelsStats:
    """Unit tests for _get_labels_stats function."""

    def test_return_labels_in_decreasing_order(self):
        """Return label indices and values sorted in decreasing order."""
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        label_array = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [2, 2, 0],
            ],
        )
        label_indices, values_returned = labels._get_labels_stats(array, label_array)

        np.testing.assert_array_equal(label_indices, np.array([0, 2, 1]))
        np.testing.assert_array_equal(values_returned, [5, 3, 1])

    def test_return_labels_in_increasing_order(self):
        """Return label indices and values sorted in increasing order."""
        array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        label_array = np.array(
            [
                [1, 0, 0],
                [2, 0, 0],
                [2, 2, 0],
            ],
        )
        label_indices, values_returned = labels._get_labels_stats(
            array,
            label_array,
            sort_decreasing=False,
        )

        np.testing.assert_array_equal(label_indices, np.array([1, 2, 0]))
        np.testing.assert_array_equal(values_returned, [1, 3, 5])


class TestVecTranslate:
    """Unit tests for _vec_translate function."""

    def test_translate_array_values_using_mapping_dictionary(self):
        """Remap array values using provided dictionary."""
        array = np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
            ],
        )
        remap = {1: 11, 2: 12, 3: 13}
        expected = np.array(
            [
                [11, 11, 11],
                [12, 12, 12],
                [13, 13, 13],
            ],
        )
        result = labels._vec_translate(array, remap)
        np.testing.assert_array_equal(result, expected)

    def test_raise_error_for_missing_key_in_remap(self):
        """Raise KeyError when remap dictionary misses array values."""
        array = np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
            ],
        )
        remap = {4: 11}
        with pytest.raises(KeyError):
            labels._vec_translate(array, remap)


class TestGetLabelsWithRequestedOccurrence:
    """Unit tests for _get_labels_with_requested_occurrence function."""

    def test_get_labels_with_specific_occurrence(self):
        """Retrieve label indices that meet occurrence criteria."""
        array = np.array(
            [
                [0, 1, 2],
                [2, 3, 3],
                [3, 0, 0],
            ],
        )
        label_indices = labels._get_labels_with_requested_occurrence(array, 1, 3)
        np.testing.assert_array_equal(label_indices, np.array([1, 2, 3]))

    def test_return_empty_array_when_no_labels_found(self):
        """Return empty array when no labels meet occurrence criteria."""
        array = np.array(
            [
                [0, 1, 2],
                [2, 3, 3],
                [3, 0, 0],
            ],
        )
        label_indices = labels._get_labels_with_requested_occurrence(array, 4, 4)
        np.testing.assert_array_equal(label_indices, np.array([]))


class TestEnsureValidLabelArr:
    """Unit tests for _ensure_valid_label_arr function."""

    def test_convert_nan_to_zero_in_label_array(self):
        """Replace NaN values with 0 in label array."""
        array = np.array([[float("nan"), 0, 1]])
        validated_array = labels._ensure_valid_label_arr(array)
        expected = np.array([[0, 0, 1]])
        np.testing.assert_array_equal(validated_array, expected)

    def test_raise_error_for_negative_values(self):
        """Raise ValueError when array contains negative values."""
        array = np.array([[-1, 0, 1]])
        with pytest.raises(ValueError):
            labels._ensure_valid_label_arr(array)

    def test_raise_error_for_non_integer_values(self):
        """Raise ValueError when array contains non-integer values."""
        array = np.array([[0.5, 0, 1]])
        with pytest.raises(ValueError):
            labels._ensure_valid_label_arr(array)


class TestEnsureValidLabelIndices:
    """Unit tests for _ensure_valid_label_indices function."""

    def test_remove_nan_and_zero_from_label_indices(self):
        """Filter out NaN and zero from label indices array."""
        label_indices = np.array([float("nan"), 0, 1, 2, 3])
        label_indices_valid = labels._ensure_valid_label_indices(label_indices)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(label_indices_valid, expected)


class TestCheckUniqueLabelIndices:
    """Unit tests for _check_unique_label_indices function."""

    def test_pass_for_unique_indices(self):
        """Accept unique label indices without error."""
        labels._check_unique_label_indices(np.array([1, 2, 3]))

    def test_raise_error_for_duplicate_indices(self):
        """Raise ValueError when label indices contain duplicates."""
        with pytest.raises(ValueError):
            labels._check_unique_label_indices(np.array([1, 2, 2]))


class TestGetNewLabelValueDict:
    """Unit tests for _get_new_label_value_dict function."""

    def test_create_mapping_for_label_values(self):
        """Create mapping dictionary for label value remapping."""
        label_indices = np.array([1, 2, 3])
        max_label = 4
        expected_dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 0,
        }
        result = labels._get_new_label_value_dict(label_indices, max_label)
        assert result == expected_dict


class TestNpRedefineLabelArray:
    """Unit tests for _np_redefine_label_array function."""

    def test_return_array_unchanged_without_label_indices(self):
        """Return original array when no label indices provided."""
        array = np.array(
            [
                [0, 1, 2],
                [2, 3, 3],
                [3, 0, 0],
            ],
        )
        array_returned = labels._np_redefine_label_array(array)
        np.testing.assert_array_equal(array_returned, array)

    def test_redefine_labels_with_filtered_indices(self):
        """Remap labels keeping only specified indices."""
        array = np.array(
            [
                [0, 1, 2],
                [2, 3, 3],
                [3, 0, 0],
            ],
        )
        label_indices = np.array([2, 3])
        expected = np.array(
            [
                [0, 0, 1],
                [1, 2, 2],
                [2, 0, 0],
            ],
        )
        result = labels._np_redefine_label_array(array, label_indices)
        np.testing.assert_array_equal(result, expected)

    def test_raise_error_for_empty_label_indices(self):
        """Raise ValueError when label indices array is empty."""
        array = np.array(
            [
                [0, 1, 2],
                [2, 3, 3],
                [3, 0, 0],
            ],
        )
        with pytest.raises(ValueError):
            labels._np_redefine_label_array(array, np.array([]))


class TestXrRedefineLabelArray:
    """Unit tests for _xr_redefine_label_array function."""

    def test_redefine_labels_in_xarray_dataarray(self):
        """Remap labels in xarray DataArray."""
        array = xr.DataArray(
            [
                [0, 1, 2],
                [2, 3, 3],
                [3, 0, 0],
            ],
        )
        label_indices = np.array([2, 3])
        expected = xr.DataArray(
            [
                [0, 0, 1],
                [1, 2, 2],
                [2, 0, 0],
            ],
        )
        result = labels._xr_redefine_label_array(array, label_indices)
        np.testing.assert_array_equal(result.values, expected.values)


class TestCheckCoreDims:
    """Unit tests for check_core_dims function."""

    def test_infer_core_dims_for_2d_dataarray(self):
        """Automatically infer core dimensions for 2D DataArray."""
        da = xr.DataArray(np.zeros((4, 5)), dims=("x", "y"))
        core_dims = check_core_dims(None, da)
        assert core_dims == ("x", "y")

    def test_accept_explicit_core_dims_for_2d(self):
        """Accept explicitly provided core dimensions for 2D DataArray."""
        da = xr.DataArray(np.zeros((4, 5)), dims=("x", "y"))
        core_dims = check_core_dims(("x", "y"), da)
        assert core_dims == ("x", "y")

    def test_accept_explicit_core_dims_for_3d(self):
        """Apply explicitly provided core dimensions for 3D DataArray."""
        da = xr.DataArray(np.zeros((2, 4, 5)), dims=("time", "x", "y"))
        core_dims = check_core_dims(("x", "y"), da)
        assert core_dims == ("x", "y")

    def test_raise_error_for_missing_core_dims_3d(self):
        """Raise error when core dimensions missing for 3D DataArray."""
        da = xr.DataArray(np.zeros((2, 4, 5)), dims=("time", "x", "y"))
        with pytest.raises(ValueError, match="ndim > 2, `core_dims` must be specified"):
            check_core_dims(None, da)

    def test_raise_error_for_too_few_core_dims(self):
        """Raise error when fewer than two core dimensions provided."""
        da = xr.DataArray(np.zeros((4, 5)), dims=("x", "y"))
        with pytest.raises(ValueError, match="exactly two dimensions"):
            check_core_dims(("x",), da)

    def test_raise_error_for_too_many_core_dims(self):
        """Raise error when more than two core dimensions provided."""
        da = xr.DataArray(np.zeros((2, 4, 5)), dims=("time", "x", "y"))
        with pytest.raises(ValueError, match="exactly two dimensions"):
            check_core_dims(("time", "x", "y"), da)

    def test_raise_error_for_non_existent_dimensions(self):
        """Raise error when core dimensions not in DataArray."""
        da = xr.DataArray(np.zeros((2, 4, 5)), dims=("time", "x", "y"))
        with pytest.raises(ValueError, match="are not all dimensions of the DataArray"):
            check_core_dims(("lat", "lon"), da)

    def test_preserve_order_of_core_dims(self):
        """Preserve user-provided order of core dimensions."""
        da = xr.DataArray(np.zeros((4, 5)), dims=("x", "y"))
        core_dims = check_core_dims(("y", "x"), da)
        assert core_dims == ("y", "x")

    def test_accept_list_input_for_core_dims(self):
        """Convert list input for core dimensions to tuple."""
        da = xr.DataArray(np.zeros((4, 5)), dims=("x", "y"))
        core_dims = check_core_dims(["x", "y"], da)
        assert isinstance(core_dims, tuple)
        assert core_dims == ("x", "y")


class TestGetDataArray:
    """Unit tests for get_data_array function."""

    def test_raise_error_for_numpy_array_input(self):
        """Raise TypeError when input is plain numpy array."""
        data = [[1, 2, 3]]
        with pytest.raises(TypeError):
            labels.get_data_array(np.array(data))

    def test_accept_dataarray_without_variable_name(self):
        """Accept DataArray input without variable name."""
        data = [[1, 2, 3]]
        array = xr.DataArray(data)
        result = labels.get_data_array(array)
        assert isinstance(result, xr.DataArray)

    def test_raise_error_for_dataarray_with_variable_name(self):
        """Raise ValueError when variable name provided for DataArray."""
        data = [[1, 2, 3]]
        array = xr.DataArray(data)
        with pytest.raises(ValueError):
            labels.get_data_array(array, "test")

    def test_extract_dataarray_from_dataset_with_variable_name(self):
        """Extract DataArray from Dataset using variable name."""
        data = [[1, 2, 3]]
        variable_name = "test"
        dataset = xr.Dataset({variable_name: xr.DataArray(data)})
        result = labels.get_data_array(dataset, variable_name)
        assert isinstance(result, xr.DataArray)

    def test_raise_error_for_dataset_with_invalid_variable(self):
        """Raise ValueError when variable name not in Dataset."""
        data = [[1, 2, 3]]
        variable_name = "test"
        dataset = xr.Dataset({variable_name: xr.DataArray(data)})
        with pytest.raises(ValueError):
            labels.get_data_array(dataset, "invalid")

    def test_raise_error_for_dataset_without_variable_name(self):
        """Raise ValueError when variable name not provided for Dataset."""
        data = [[1, 2, 3]]
        variable_name = "test"
        dataset = xr.Dataset({variable_name: xr.DataArray(data)})
        with pytest.raises(ValueError):
            labels.get_data_array(dataset)


class TestGetLabels:
    """Unit tests for _get_labels function."""

    def test_get_labels_with_default_arguments(self):
        """Apply labeling with default arguments (decreasing area order)."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        labels_array, n_labels, values = labels._get_labels(array)
        expected = np.array(
            [
                [3, 3, 0, 0, 2],
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 4, 0],
            ],
        )
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 4
        np.testing.assert_array_equal(values, np.array([3, 2, 2, 1]))

    def test_filter_labels_by_value_thresholds(self):
        """Filter labels by minimum and maximum value thresholds."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        labels_array, n_labels, values = labels._get_labels(
            array,
            min_value_threshold=3,
            max_value_threshold=6,
        )
        expected = np.array(
            [
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
        )
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 2
        np.testing.assert_array_equal(values, np.array([2, 2]))

    def test_return_no_labels_when_all_filtered_by_value(self):
        """Return zero array when value filtering removes all labels."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        labels_array, n_labels, values = labels._get_labels(array, max_value_threshold=0)
        expected = np.zeros(array.shape)
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 0
        np.testing.assert_array_equal(values, np.array([]))

    def test_filter_labels_by_area_thresholds(self):
        """Filter labels by minimum and maximum area thresholds."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        labels_array, n_labels, values = labels._get_labels(
            array,
            min_area_threshold=2,
            max_area_threshold=2,
        )
        expected = np.array(
            [
                [2, 2, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 2
        np.testing.assert_array_equal(values, np.array([2, 2]))

    def test_return_no_labels_when_all_filtered_by_area(self):
        """Return zero array when area filtering removes all labels."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        labels_array, n_labels, values = labels._get_labels(array, min_area_threshold=4)
        expected = np.zeros(array.shape)
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 0
        np.testing.assert_array_equal(values, np.array([]))

    def test_apply_custom_footprint_for_connectivity(self):
        """Apply custom footprint to define label connectivity."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        footprint = np.array([[1, 1], [1, 1]])
        labels_array, n_labels, values = labels._get_labels(array, footprint=footprint)
        expected = np.array(
            [
                [3, 3, 0, 0, 2],
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0],
            ],
        )
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 3
        np.testing.assert_array_equal(values, np.array([4, 2, 2]))

    def test_sort_labels_by_mean_value_ascending(self):
        """Sort labels by mean value in ascending order."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )
        labels_array, n_labels, values = labels._get_labels(
            array,
            sort_by="mean",
            sort_decreasing=False,
        )
        expected = np.array(
            [
                [1, 1, 0, 0, 2],
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0],
                [3, 3, 0, 4, 0],
            ],
        )
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 4
        np.testing.assert_allclose(values, np.array([1.5, 3.5, 6, 8]))

    def test_custom_sorting_with_user_function(self):
        """Apply custom sorting function with all filtering parameters."""
        _ = float("nan")
        array = np.array(
            [
                [1, 2, _, _, 3],
                [_, _, _, _, 4],
                [_, _, _, _, _],
                [5, _, _, _, _],
                [6, 7, _, 8, _],
            ],
        )

        def my_sort_function(arr):
            return np.mean(arr)

        footprint = np.array([[1, 1], [1, 1]])
        labels_array, n_labels, values = labels._get_labels(
            array,
            min_value_threshold=3,
            max_value_threshold=8,
            min_area_threshold=2,
            max_area_threshold=4,
            footprint=footprint,
            sort_by=my_sort_function,
            sort_decreasing=False,
            labeled_comprehension_kwargs={"out_dtype": np.float16},
        )
        expected = np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 2, 0, 2, 0],
            ],
            dtype="float",
        )
        np.testing.assert_array_equal(labels_array, expected)
        assert n_labels == 2
        np.testing.assert_allclose(values, np.array([3.5, 6.5]))
