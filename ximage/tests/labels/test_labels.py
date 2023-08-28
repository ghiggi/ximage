import numpy as np
import pytest
import xarray as xr

from pytest import apply_to_all_array_types

from ximage.labels import labels


# Tests for public functions ###################################################


def test_get_label_indices():
    """Test label indices retrieval"""

    array = [float("nan"), 0, 1, 1, 2.001, 4, 1]
    label_indices = np.array([1, 2, 4])

    def check(array):
        label_indices_returned = labels.get_label_indices(array)
        assert np.array_equal(label_indices_returned, label_indices)
        assert label_indices_returned.dtype == np.int64 or label_indices_returned.dtype == np.int32

    apply_to_all_array_types(check, array)


def test_redefine_label_array():
    """Test label redefinition"""

    array = [3, 3, 4, 6]
    label_indices = np.array([3, 4])
    label_indices_with_0 = np.array([0, 3, 4])
    label_indices_with_duplicate = np.array([3, 3, 4])
    redefined_array = [1, 1, 2, 0]
    redefined_array_default = [1, 1, 2, 3]

    def check(array):
        assert np.array_equal(labels.redefine_label_array(array, label_indices), redefined_array)

        # Test default labels
        assert np.array_equal(labels.redefine_label_array(array), redefined_array_default)

        # Test invalid labels
        assert np.array_equal(
            labels.redefine_label_array(array, label_indices_with_0), redefined_array
        )

        with pytest.raises(ValueError):
            labels.redefine_label_array(array, label_indices_with_duplicate)

    apply_to_all_array_types(check, array)

    # Test invalid array type
    with pytest.raises(TypeError):
        labels.redefine_label_array(array)


def test_labels():
    """Test labels. See test_get_labels for extensive tests of all arguments."""

    _ = float("nan")
    array = np.array(
        [
            [1, 2, _, _, 3],
            [_, _, _, _, 4],
            [_, _, _, _, _],
            [5, _, _, _, _],
            [6, 7, _, 8, _],
        ]
    )

    # Try wrong array type
    with pytest.raises(TypeError):
        labels.label(array)

    # Check that all arguments are passed correctly to xr_get_labels
    data_array = xr.DataArray(array)
    min_value = 3
    max_value = 8
    min_area = 2
    max_area = 4
    footprint = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )
    sort_by = "mean"
    sort_decreasing = False
    data_array_returned = labels.label(
        data_array,
        min_value_threshold=min_value,
        max_value_threshold=max_value,
        min_area_threshold=min_area,
        max_area_threshold=max_area,
        footprint=footprint,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
    )
    labels_array_expected = np.array(
        [
            [_, _, _, _, 1],
            [_, _, _, _, 1],
            [_, _, _, _, _],
            [2, _, _, _, _],
            [2, 2, _, 2, _],
        ]
    )
    assert np.array_equal(
        data_array_returned.coords["label"], labels_array_expected, equal_nan=True
    )

    # Test with no label returned
    max_value = 0
    with pytest.raises(ValueError):
        data_array_returned = labels.label(data_array, max_value_threshold=max_value)

    # Test with Dataset
    variable_name = "data"
    dataset = xr.Dataset({variable_name: data_array})
    dataset_returned = labels.label(dataset, variable_name)
    labels_array_expected = np.array(
        [
            [3, 3, _, _, 2],
            [_, _, _, _, 2],
            [_, _, _, _, _],
            [1, _, _, _, _],
            [1, 1, _, 4, _],
        ]
    )
    assert np.array_equal(dataset_returned.coords["label"], labels_array_expected, equal_nan=True)


def test_highlight_label():
    """Test highlight_label"""

    _ = float("nan")
    labels_array = xr.DataArray(
        [
            [_, 1, 2],
            [3, _, 2],
            [3, 3, _],
        ]
    )
    label_name = "test_label"
    label_id = 2
    dataset = xr.Dataset({label_name: labels_array})
    dataset_returned = labels.highlight_label(dataset, label_name, label_id)
    labels_array_expected = np.array(
        [
            [0, 0, 2],
            [0, 0, 2],
            [0, 0, 0],
        ]
    )
    assert np.array_equal(dataset_returned[label_name], labels_array_expected)


# Tests for internal functions #################################################


def test_mask_buffer():
    """Test mask dilation"""

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

    mask_dilated_target = footprint

    # No dilation
    assert np.array_equal(labels._mask_buffer(mask_ini, None), mask_ini)
    assert np.array_equal(labels._mask_buffer(mask_ini, 0), mask_ini)

    # Dilation of 1
    assert np.array_equal(labels._mask_buffer(mask_ini, 1), mask_dilated_target)

    # Dilation with mask
    assert np.array_equal(labels._mask_buffer(mask_ini, footprint), mask_dilated_target)

    # Errors
    with pytest.raises(TypeError):
        labels._mask_buffer(mask_ini, "invalid type")

    with pytest.raises(ValueError):
        labels._mask_buffer(mask_ini, -1)

    with pytest.raises(ValueError):
        non_2D_footprint = np.array([1, 1, 1])
        labels._mask_buffer(mask_ini, non_2D_footprint)


def test_check_array():
    """Test array dimensions checks and conversion to np.ndarray"""

    # Check error if non-2D
    with pytest.raises(ValueError):
        labels._check_array(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        labels._check_array(np.ones((3, 3, 3)))

    # Check error if dimension of size 0
    with pytest.raises(ValueError):
        labels._check_array(np.ones((0, 3)))

    # Check conversion to np.ndarray
    array = np.ones((3, 3))

    def check(array):
        checked_array = labels._check_array(array)
        assert isinstance(checked_array, np.ndarray)
        assert np.array_equal(checked_array, np.asanyarray(array))

    apply_to_all_array_types(check, array)


def test_no_labels_result():
    """Test default labels for array without labels"""

    array = np.ones((2, 3))
    labels_target = np.zeros(array.shape)

    labels_result, n_labels, values = labels._no_labels_result(array)
    assert np.array_equal(labels_result, labels_target)
    assert n_labels == 0
    assert values == []


def test_check_sort_by_and_stats():
    """Tests for sort_by argument"""

    # Check if arg is a string or function
    with pytest.raises(TypeError):
        labels._check_sort_by(None)

    with pytest.raises(TypeError):
        labels._check_stats(None)

    # Check if arg is not a valid string
    with pytest.raises(ValueError):
        labels._check_sort_by("invalid string")

    with pytest.raises(ValueError):
        labels._check_stats("invalid string")

    # Check stats returned
    stats = "mean"
    assert labels._check_stats(stats) == stats


def test_get_label_value_stats():
    """Test _get_label_value_stats"""

    array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    label_array = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [2, 2, 0],
        ]
    )

    values_area = [5, 1, 3]

    # With default labels indices (all) and stats (area)
    values = labels._get_label_value_stats(array, label_array)
    assert np.array_equal(values, values_area)

    # Custom stats function
    def stats(*args, **kwargs):
        return 42

    values_area = [42, 42, 42]

    values = labels._get_label_value_stats(array, label_array, stats=stats)
    assert np.array_equal(values, values_area)


def test_get_labels_stats():
    """Test _get_labels_stats"""

    array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    label_array = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [2, 2, 0],
        ]
    )

    values_area = [5, 1, 3]

    # Test result returned in decreasing order
    label_indices, values_returned = labels._get_labels_stats(array, label_array)
    assert np.array_equal(label_indices, np.array([0, 2, 1]))
    assert np.array_equal(values_returned, [5, 3, 1])

    # Test result returned in increasing order
    label_indices, values_returned = labels._get_labels_stats(
        array, label_array, sort_decreasing=False
    )
    assert np.array_equal(label_indices, np.array([1, 2, 0]))
    assert np.array_equal(values_returned, [1, 3, 5])


def test_vec_translate():
    """Test _vec_translate"""

    array = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )

    remap = {
        1: 11,
        2: 12,
        3: 13,
    }

    array_remapped = np.array(
        [
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
        ]
    )

    array_returned = labels._vec_translate(array, remap)
    assert np.array_equal(array_returned, array_remapped)

    # Check keys that are not in the array or wrong key types
    for key in [4, None, "1", np.nan]:
        remap = {key: 11}
        with pytest.raises(KeyError):
            labels._vec_translate(array, remap)


def test_get_labels_with_requested_occurrence():
    """Test _get_labels_with_requested_occurrence"""

    array = np.array(
        [
            [0, 1, 2],
            [2, 3, 3],
            [3, 0, 0],
        ]
    )

    label_indices = labels._get_labels_with_requested_occurrence(array, 1, 3)
    assert np.array_equal(label_indices, np.array([1, 2, 3]))

    # Test no labels found
    label_indices = labels._get_labels_with_requested_occurrence(array, 4, 4)
    assert np.array_equal(label_indices, np.array([]))


def test_ensure_valid_label_arr():
    """Test _ensure_valid_label_arr"""

    array = np.array([[float("nan"), 0, 1]])
    validated_array = labels._ensure_valid_label_arr(array)
    assert np.array_equal(validated_array, np.array([[0, 0, 1]]))

    # Test invalid values (negative, non-integer)
    invalid_values = [-1, 0.5]
    for invalid_value in invalid_values:
        array = np.array([[invalid_value, 0, 1]])
        with pytest.raises(ValueError):
            labels._ensure_valid_label_arr(array)


def test_ensure_valid_label_indices():
    """Test _ensure_valid_label_indices"""

    label_indices = np.array([float("nan"), 0, 1, 2, 3])
    label_indices_valid = labels._ensure_valid_label_indices(label_indices)
    assert np.array_equal(label_indices_valid, np.array([1, 2, 3]))


def test_check_unique_label_indices():
    """Test _check_unique_label_indices"""

    labels._check_unique_label_indices(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        labels._check_unique_label_indices(np.array([1, 2, 2]))


def test_get_new_label_value_dict():
    """Test _get_new_label_value_dict"""

    label_indices = np.array([1, 2, 3])
    max_label = 4
    expected_value_dict = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 0,
    }

    new_label_value_dict = labels._get_new_label_value_dict(label_indices, max_label)
    assert new_label_value_dict == expected_value_dict


def test_np_redefine_label_array():
    """Test _np_redefine_label_array"""

    array = np.array(
        [
            [0, 1, 2],
            [2, 3, 3],
            [3, 0, 0],
        ]
    )

    # Call without providing label_indices
    array_returned = labels._np_redefine_label_array(array)
    assert np.array_equal(array_returned, array)

    # Call with filtering label_indices
    label_indices = np.array([2, 3])
    array_expected = np.array(
        [
            [0, 0, 1],
            [1, 2, 2],
            [2, 0, 0],
        ]
    )
    array_returned = labels._np_redefine_label_array(array, label_indices)
    assert np.array_equal(array_returned, array_expected)

    # Check exception raised if label_indices is empty
    with pytest.raises(ValueError):
        labels._np_redefine_label_array(array, np.array([]))


def test_xr_redefine_label_array():
    """Test _xr_redefine_label_array"""

    array = xr.DataArray(
        [
            [0, 1, 2],
            [2, 3, 3],
            [3, 0, 0],
        ]
    )
    label_indices = np.array([2, 3])
    array_expected = xr.DataArray(
        [
            [0, 0, 1],
            [1, 2, 2],
            [2, 0, 0],
        ]
    )

    array_returned = labels._xr_redefine_label_array(array, label_indices)
    assert np.array_equal(array_returned, array_expected)


def test_check_xr_obj():
    """Test _check_xr_obj"""

    data = [[1, 2, 3]]

    # Check invalid type
    with pytest.raises(TypeError):
        labels._check_xr_obj(np.array(data))

    # Check DataArray
    array = xr.DataArray(data)
    labels._check_xr_obj(array)

    # Check DataArray with provided variable name
    with pytest.raises(ValueError):
        labels._check_xr_obj(array, "test")

    # Check Dataset
    variable_name = "test"
    dataset = xr.Dataset({variable_name: array})
    labels._check_xr_obj(dataset, variable_name)

    # Check Dataset with invalid variable name
    with pytest.raises(ValueError):
        labels._check_xr_obj(dataset, "invalid")

    # Check Dataset with no variable name
    with pytest.raises(ValueError):
        labels._check_xr_obj(dataset)


def test_get_labels():
    """Test _get_labels"""

    _ = float("nan")
    array = np.array(
        [
            [1, 2, _, _, 3],
            [_, _, _, _, 4],
            [_, _, _, _, _],
            [5, _, _, _, _],
            [6, 7, _, 8, _],
        ]
    )

    # Test with all default arguments (labels ordered by decreasing area)
    labels_array_returned, n_labels_returned, values_returned = labels._get_labels(array)
    labels_array_expected = np.array(
        [
            [3, 3, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 4, 0],
        ]
    )
    n_labels_expected = 4
    values_expected = np.array([3, 2, 2, 1])
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)

    # Test with filtered values
    min_value = 3
    max_value = 6
    labels_array_returned, n_labels_returned, values_returned = labels._get_labels(
        array, min_value_threshold=min_value, max_value_threshold=max_value
    )
    labels_array_expected = np.array(
        [
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    )
    n_labels_expected = 2
    values_expected = np.array([2, 2])
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)

    # Test with no label returned (due to value filtering)
    max_value = 0
    labels_array_returned, n_labels_returned, values_returned = labels._get_labels(
        array, max_value_threshold=max_value
    )
    labels_array_expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    n_labels_expected = 0
    values_expected = np.array([])
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)

    # Test with filtered area
    min_area = 2
    max_area = 2
    labels_array_returned, n_labels_returned, values_returned = labels._get_labels(
        array, min_area_threshold=min_area, max_area_threshold=max_area
    )
    labels_array_expected = np.array(
        [
            [2, 2, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    n_labels_expected = 2
    values_expected = np.array([2, 2])
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)

    # Test with no label returned (due to area filtering)
    min_area = 4
    labels_array_returned, n_labels_returned, values_returned = labels._get_labels(
        array, min_area_threshold=min_area
    )
    labels_array_expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    n_labels_expected = 0
    values_expected = np.array([])
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)

    # Test non-default footprint
    footprint = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )
    labels_array_returned, n_labels_returned, values_returned = labels._get_labels(
        array, footprint=footprint
    )
    labels_array_expected = np.array(
        [
            [3, 3, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0],
        ]
    )
    n_labels_expected = 3
    values_expected = np.array([4, 2, 2])
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)

    # Test other sorting method
    sort_by = "mean"
    sort_decreasing = False
    labels_array_returned, n_labels_returned, values_returned = labels._get_labels(
        array, sort_by=sort_by, sort_decreasing=sort_decreasing
    )
    labels_array_expected = np.array(
        [
            [1, 1, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [3, 3, 0, 4, 0],
        ]
    )
    n_labels_expected = 4
    values_expected = np.array([1.5, 3.5, 6, 8])
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)


def test_xr_get_labels():
    """Test _xr_get_labels. See test_get_labels for extensive tests of all arguments."""

    _ = float("nan")
    array = np.array(
        [
            [1, 2, _, _, 3],
            [_, _, _, _, 4],
            [_, _, _, _, _],
            [5, _, _, _, _],
            [6, 7, _, 8, _],
        ]
    )

    # Try wrong array type
    with pytest.raises(TypeError):
        labels._xr_get_labels(array)

    # Check that all arguments are passed correctly to _get_labels
    array = xr.DataArray(array)
    min_value = 3
    max_value = 8
    min_area = 2
    max_area = 4
    footprint = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )
    sort_by = lambda array: np.mean(array)
    sort_decreasing = False
    labeled_comprehension_kwargs = {
        "out_dtype": np.float16,
    }
    labels_array_returned, n_labels_returned, values_returned = labels._xr_get_labels(
        array,
        min_value_threshold=min_value,
        max_value_threshold=max_value,
        min_area_threshold=min_area,
        max_area_threshold=max_area,
        footprint=footprint,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
        labeled_comprehension_kwargs=labeled_comprehension_kwargs,
    )
    labels_array_expected = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [2, 2, 0, 2, 0],
        ]
    )
    n_labels_expected = 2
    values_expected = np.array([3.5, 6.5])
    assert type(labels_array_returned) == xr.DataArray
    assert np.array_equal(labels_array_returned, labels_array_expected)
    assert n_labels_returned == n_labels_expected
    assert np.array_equal(values_returned, values_expected)
    assert values_returned.dtype == np.float16
