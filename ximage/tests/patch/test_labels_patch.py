import numpy as np
import pytest
import xarray as xr

from pytest import apply_to_all_array_types, SaneEqualityArray

from ximage.patch import labels_patch


# Utils functions ##############################################################


def check_slices_and_patches(
    args, kwargs, variable_array, variable_name, labels_array, label_name, expected_slices_dict
):
    """Function for testing get_patches_from_labels and get_patches_isel_dict_from_labels.

    Calls the two methods with "args" and "kwargs", checks that the returned slices are the same as the expected ones,
    and checks that the patches have the expected sliced variable and labels arrays.
    """

    # Get slices
    returned_slices_dict = labels_patch.get_patches_isel_dict_from_labels(*args, **kwargs)

    # Compare slices
    assert (
        returned_slices_dict == expected_slices_dict
    ), f"Returned slices are not the same as the expected ones. Returned: {returned_slices_dict}, expected: {expected_slices_dict}"

    # Get patches (variable + labels)
    returned_variable_dict = {}
    returned_labels_dict = {}

    for label, patch in labels_patch.get_patches_from_labels(*args, **kwargs):
        if label not in returned_variable_dict:
            returned_variable_dict[label] = []
            returned_labels_dict[label] = []

        returned_variable_dict[label].append(SaneEqualityArray(patch[variable_name]))
        returned_labels_dict[label].append(SaneEqualityArray(patch[label_name]))

    # Build expected patches
    expected_variables_dict = {}
    expected_labels_dict = {}

    for label in expected_slices_dict:
        expected_variables_dict[label] = []
        expected_labels_dict[label] = []

        for slices in expected_slices_dict[label]:
            sliced_variable_array = variable_array[slices["x"], slices["y"]]
            sliced_labels_array = labels_array[slices["x"], slices["y"]].copy()
            sliced_labels_array[sliced_labels_array != label] = 0
            expected_variables_dict[label].append(SaneEqualityArray(sliced_variable_array))
            expected_labels_dict[label].append(SaneEqualityArray(sliced_labels_array))

    # Compare patches (deep equality possible thanks to SaneEqualityArray)
    assert (
        returned_variable_dict == expected_variables_dict
    ), f"Returned variable patches are not the same as the expected ones. Returned: {returned_variable_dict}, expected: {expected_variables_dict}"
    assert (
        returned_labels_dict == expected_labels_dict
    ), f"Returned labels patches are not the same as the expected ones. Returned: {returned_labels_dict}, expected: {expected_labels_dict}"


# Tests for public functions ###################################################


def test_find_point():
    """Test find_point"""

    _ = float("nan")

    array = np.array(
        [
            [_, _, _],
            [0, 1, _],
            [4, 2, 5],
        ]
    )

    assert labels_patch.find_point(array, "min") == (1, 0)
    assert labels_patch.find_point(array, "max") == (2, 2)
    assert labels_patch.find_point(array, "centroid") == (1, 1)
    assert labels_patch.find_point(array, "center_of_mass") == (2, 1)

    random_point = labels_patch.find_point(array, "random")
    assert random_point[0] in [0, 1, 2]
    assert random_point[1] in [0, 1, 2]

    def centered_on_1_2(array):
        if np.all(np.isnan(array)):
            return None
        return (1, 2)

    assert labels_patch.find_point(array, centered_on_1_2) == (1, 2)


def test_get_patches_and_isel_dict_from_labels(monkeypatch):
    """Test get_patches_from_labels and get_patches_isel_dict_from_labels"""

    # Create variable array
    # [[ 0,  1...,  9]
    #  [10, 11..., 19]
    #  ...
    #  [90, 91..., 99]]
    shape = (10,) * 2
    dims = ["x", "y"]

    variable_array = np.reshape(np.arange(shape[0] * shape[1], dtype=float), shape)

    # Create label array (single label on diagonal)
    # [[0, 0, 0, ..., 0, 0]
    #  [0, 1, 0, ..., 0, 0]
    #  [0, 0, 1, ..., 0, 0]
    #  ...
    #  [0, 0, 0, ..., 1, 0]
    #  [0, 0, 0, ..., 0, 0]]
    diag_label_array = np.ones(shape[0])
    diag_label_array[0] = 0
    diag_label_array[-1] = 0

    label_array = np.diag(diag_label_array)

    # Create Dataset with variable and label
    variable_data_array = xr.DataArray(variable_array, dims=dims)
    label_data_array = xr.DataArray(label_array, dims=dims)

    variable_name = "variable"
    label_name = "label"

    dataset = xr.Dataset({variable_name: variable_data_array, label_name: label_data_array})

    # Test with default parameters: centered_on="max", no partitioning method (therefore only one patch)
    patch_size = 3
    args = (dataset, label_name, patch_size)
    kwargs = {
        "variable": variable_name,
    }
    expected_slices = {
        1: [
            {"x": slice(7, 10), "y": slice(7, 10)},  # Bottom right corner, centered on max
        ],
    }
    check_slices_and_patches(
        args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices
    )

    # Test verbose and debug arguments
    kwargs = {
        "variable": variable_name,
        "verbose": True,
        "debug": True,
    }
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)  # Prevent showing plots
    check_slices_and_patches(
        args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices
    )

    # Test with tiling parameters
    kwargs = {
        "variable": variable_name,
        "centered_on": "label_bbox",
        "partitioning_method": "tiling",
    }
    expected_slices = {
        1: [
            {"x": slice(1, 4), "y": slice(1, 4)},
            {"x": slice(4, 7), "y": slice(4, 7)},
            {"x": slice(5, 8), "y": slice(5, 8)},
            # From (4, 7)(6, 9) and (6, 9)(4, 7) that got recentered
            {"x": slice(6, 9), "y": slice(6, 9)},
        ],
    }
    check_slices_and_patches(
        args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices
    )

    # Test with two labels
    # [[0, 2, 0, ..., 0, 0]
    #  [0, 1, 2, ..., 0, 0]
    #  [0, 0, 1, ..., 0, 0]
    #  ...
    #  [0, 0, 0, ..., 1, 2]
    #  [0, 0, 0, ..., 0, 0]]
    # Modify label_array in-place. Reflects in Dataset.
    label_array[:] = label_array + np.diag(np.ones(shape[0] - 1) * 2, k=1)
    expected_slices[2] = [
        {"x": slice(0, 3), "y": slice(1, 4)},
        {"x": slice(3, 6), "y": slice(4, 7)},
        {"x": slice(6, 9), "y": slice(7, 10)},
    ]
    check_slices_and_patches(
        args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices
    )

    # Test n_patches
    # kwargs["n_patches"] = 2
    # expected_slices_cut = {
    #     1: expected_slices[1][:2],
    #     2: expected_slices[2][:2],
    #         }
    # check_slices_and_patches(args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices_cut)
    # del kwargs["n_patches"]

    # Test n_partitions_per_label (total number of partitions before filtering those that do not contain labels)
    expected_slices_cut = {
        1: expected_slices[1][:2],
        2: expected_slices[2][:2],
    }
    kwargs["n_partitions_per_label"] = 5  # over 9 possible partitions in this case
    check_slices_and_patches(
        args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices_cut
    )

    # Invalid label request
    kwargs = {
        "variable": variable_name,
        "n_labels": 2,
        "labels_id": 1,
    }
    with pytest.raises(ValueError):
        check_slices_and_patches(
            args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices
        )

    # Test with nan regions
    nan_region = [slice(5, 10), slice(5, 10)]
    variable_array[nan_region[0], nan_region[1]] = np.nan
    expected_slices_cut = {
        1: expected_slices[1][:2],
        2: expected_slices[2][:2],
    }
    kwargs = {
        "variable": variable_name,
        "centered_on": "centroid",
        "partitioning_method": "tiling",
    }
    check_slices_and_patches(
        args, kwargs, variable_array, variable_name, label_array, label_name, expected_slices_cut
    )


# Tests for internal functions #################################################


def test_check_label_arr():
    """Test _check_label_arr"""

    _ = np.nan

    # Check conversion of 0 to nan
    array = np.array(
        [
            [_, 0, 1],
            [2, 3, 4],
        ]
    )

    expected_array = np.array(
        [
            [_, _, 1],
            [2, 3, 4],
        ]
    )

    def check(array):
        returned_array = labels_patch._check_label_arr(array)
        assert np.array_equal(returned_array, expected_array, equal_nan=True)
        assert isinstance(returned_array, np.ndarray)

    apply_to_all_array_types(check, array)

    # Check non integers
    array = np.array(
        [
            [_, 0, 0.1],
        ]
    )
    with pytest.raises(ValueError):
        labels_patch._check_label_arr(array)

    # Check non-positive integers
    array = np.array(
        [
            [_, 0, -1],
        ]
    )
    with pytest.raises(ValueError):
        labels_patch._check_label_arr(array)


def test_check_labels_id():
    """Test _check_labels_id"""

    _ = np.nan

    label_array = np.array(
        [
            [_, _, 1],
            [2, 3, 5],
        ]
    )

    # Check without input labels_id
    expected_labels_id = np.array([1, 2, 3, 5])
    returned_labels_id = labels_patch._check_labels_id(None, label_array)
    assert np.array_equal(returned_labels_id, expected_labels_id)

    # Check integer labels_id
    labels_id = 3
    expected_labels_id = np.array([3])
    returned_labels_id = labels_patch._check_labels_id(labels_id, label_array)
    assert np.array_equal(returned_labels_id, expected_labels_id)
    assert isinstance(returned_labels_id, np.ndarray)

    # Check list labels_id
    labels_id = [3, 5]
    expected_labels_id = np.array([3, 5])
    returned_labels_id = labels_patch._check_labels_id(labels_id, label_array)
    assert np.array_equal(returned_labels_id, expected_labels_id)
    assert isinstance(returned_labels_id, np.ndarray)

    # Check ndarray labels_id
    labels_id = np.array([3, 5])
    returned_labels_id = labels_patch._check_labels_id(labels_id, label_array)
    assert np.array_equal(returned_labels_id, expected_labels_id)
    assert isinstance(returned_labels_id, np.ndarray)

    # Check conversion in int dtype
    labels_id = np.array([3, 5], dtype=np.float32)
    returned_labels_id = labels_patch._check_labels_id(labels_id, label_array)
    assert returned_labels_id.dtype == np.int64 or returned_labels_id.dtype == np.int32

    # Check invalid labels_id: 0, non-positive, non-integer, not in label_array
    for label in [0, -1, 0.1, 4]:
        labels_id = [label]
        with pytest.raises(ValueError):
            labels_patch._check_labels_id(labels_id, label_array)

    # Check empty labels_id
    labels_id = []
    with pytest.raises(ValueError):
        labels_patch._check_labels_id(labels_id, label_array)

    # Check wrong labels_id type
    for labels_id in [1.0, "1", (1,)]:
        with pytest.raises(TypeError):
            labels_patch._check_labels_id(labels_id, label_array)


def test_check_n_patches_per_partition():
    """Test _check_n_patches_per_partition"""

    # Check valid n_patches_per_partition
    centered_on = "max"
    assert labels_patch._check_n_patches_per_partition(1, centered_on) == 1

    centered_on = "random"
    assert labels_patch._check_n_patches_per_partition(2, centered_on) == 2

    # Check invalid n_patches_per_partition
    centered_on = "max"
    for n_patches_per_partition in [0, -1, 2]:
        with pytest.raises(ValueError):
            labels_patch._check_n_patches_per_partition(n_patches_per_partition, centered_on)


def test_check_n_patches_per_label():
    """Test _check_n_patches_per_label"""

    # Valid n_patches_per_label
    n_patches_per_partition = 10
    for n_patches_per_label in [10, 11]:
        assert (
            labels_patch._check_n_patches_per_label(n_patches_per_label, n_patches_per_partition)
            == n_patches_per_label
        )

    # Invalid n_patches_per_label
    for n_patches_per_label in [9, 8]:
        with pytest.raises(ValueError):
            labels_patch._check_n_patches_per_label(n_patches_per_label, n_patches_per_partition)


def test_check_callable_centered_on():
    """Test _check_callable_centered_on"""

    # Check valid centered_on
    def centered_on_1(array):
        if np.all(np.isnan(array)):
            return None
        return (1,) * len(array.shape)

    labels_patch._check_callable_centered_on(centered_on_1)

    # Check invalid return type
    for returned_point in [1, [1, 2]]:

        def centered_on_2(array):
            if np.all(np.isnan(array)):
                return None
            return returned_point

        with pytest.raises(ValueError):
            labels_patch._check_callable_centered_on(centered_on_2)

    # Check invalid dimension
    def centered_on_3(array):
        if np.all(np.isnan(array)):
            return None
        return (1,) * (len(array.shape) + 1)

    with pytest.raises(ValueError):
        labels_patch._check_callable_centered_on(centered_on_3)

    # Check coordinates outside shape (too small)
    def centered_on_4(array):
        if np.all(np.isnan(array)):
            return None
        return (-1,) * len(array.shape)

    with pytest.raises(ValueError):
        labels_patch._check_callable_centered_on(centered_on_4)

    # Check coordinates outside shape (too large)
    def centered_on_5(array):
        if np.all(np.isnan(array)):
            return None
        coordinate = list(array.shape)
        for i in range(len(coordinate)):
            coordinate[i] += 1
        return tuple(coordinate)

    with pytest.raises(ValueError):
        labels_patch._check_callable_centered_on(centered_on_5)

    # Check nan in return
    def centered_on_6(array):
        if np.all(np.isnan(array)):
            return None
        coordinate = [1.0] * len(array.shape)
        coordinate[0] = np.nan
        return tuple(coordinate)

    with pytest.raises(ValueError):
        labels_patch._check_callable_centered_on(centered_on_6)

    # Check function that cannot deal with nan
    def centered_on_7(array):
        return (int(array.flatten()[0]),) * len(array.shape)

    with pytest.raises(ValueError):
        labels_patch._check_callable_centered_on(centered_on_7)

    # Check function that does not return None if array is nan
    def centered_on_8(array):
        return (1,) * len(array.shape)

    with pytest.raises(ValueError):
        labels_patch._check_callable_centered_on(centered_on_8)


def test_check_centered_on():
    """Test _check_centered_on"""

    # Check valid centered_on
    def valid_callable(array):
        if np.all(np.isnan(array)):
            return None
        return (1,) * len(array.shape)

    valid_centered_on = [
        "max",
        "min",
        "centroid",
        "center_of_mass",
        "random",
        "label_bbox",
        valid_callable,
    ]

    for centered_on in valid_centered_on:
        assert labels_patch._check_centered_on(centered_on) == centered_on

    # Check invalid centered_on
    with pytest.raises(TypeError):
        labels_patch._check_centered_on(1)

    def invalid_callable(array):
        return (1,) * len(array.shape)

    invalid_centered_on = [
        "invalid",
        invalid_callable,
    ]

    for centered_on in invalid_centered_on:
        with pytest.raises(ValueError):
            labels_patch._check_centered_on(centered_on)


def test_get_variable_arr():
    """Test _get_variable_arr"""

    array = np.random.rand(2, 3)
    variable_name = None
    centered_on = None

    # Check with DataArray
    data_array = xr.DataArray(array)
    returned_variable_array = labels_patch._get_variable_arr(data_array, variable_name, centered_on)
    assert np.array_equal(returned_variable_array, array)

    # Check with Dataset
    variable_name = "variable"
    dataset = xr.Dataset({variable_name: data_array})
    returned_variable_array = labels_patch._get_variable_arr(dataset, None, centered_on)
    assert returned_variable_array is None

    returned_variable_array = labels_patch._get_variable_arr(dataset, variable_name, centered_on)
    assert np.array_equal(returned_variable_array, array)

    # Check unspecified variable with centered_on
    centered_on = "max"
    with pytest.raises(ValueError):
        labels_patch._get_variable_arr(dataset, None, centered_on)


def test_check_variable_arr():
    """Test _check_variable_arr"""

    shape = (2, 3)

    # Check valid variable array (same shape as label arary)
    variable_array = np.random.rand(*shape)
    label_array = np.random.randint(1, 5, shape)
    assert labels_patch._check_variable_arr(variable_array, label_array) is variable_array

    # Check invalid variable array (different shape as label arary)
    variable_array = np.random.rand(3, 3)
    with pytest.raises(ValueError):
        labels_patch._check_variable_arr(variable_array, label_array)


def test_get_point_centroid():
    """Test _get_point_centroid"""

    shape = (4, 5)
    array = np.random.rand(*shape)
    expected_centroid = tuple(np.array(shape) / 2)
    assert labels_patch._get_point_centroid(array) == expected_centroid

    # Check None return if only nan in array
    array = np.full(shape, np.nan)
    assert labels_patch._get_point_centroid(array) is None


def test_get_point_random():
    """Test _get_point_random"""

    shape = (4, 5)
    array = np.random.rand(*shape)
    returned_point = labels_patch._get_point_random(array)
    assert len(returned_point) == len(shape)
    for max_value, coordinate in zip(shape, returned_point):
        assert 0 <= coordinate
        assert coordinate < max_value

    # Check None return if only nan in array
    array = np.full(shape, np.nan)
    assert labels_patch._get_point_random(array) is None


def test_get_point_with_max_value():
    """Test _get_point_with_max_value"""

    shape = (4, 5)
    array = np.random.rand(*shape)
    max_value = np.max(array)

    returned_point = labels_patch._get_point_with_max_value(array)
    assert array[returned_point] == max_value

    # Check None return if only nan in array
    array = np.full(shape, np.nan)
    assert labels_patch._get_point_with_max_value(array) is None


def test_get_point_with_min_value():
    """Test _get_point_with_min_value"""

    shape = (4, 5)
    array = np.random.rand(*shape)
    min_value = np.min(array)

    returned_point = labels_patch._get_point_with_min_value(array)
    assert array[returned_point] == min_value

    # Check None return if only nan in array
    array = np.full(shape, np.nan)
    assert labels_patch._get_point_with_min_value(array) is None


def test_get_point_center_of_mass():
    """Test _get_point_center_of_mass"""

    _ = np.nan

    array = np.array(
        [
            [0, 0, 0, _],
            [1, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, _],
        ]
    )
    # Only location of non-nan values matter

    expected_point = (1.5, 18 / 14)  # 4*0 + 4*1 + 4*2 + 2*3 = 18, over 14 non-nan values
    returned_point = labels_patch._get_point_center_of_mass(array, integer_index=False)
    assert returned_point == expected_point

    # Test with integer index (default)
    expected_point = (2, 1)
    returned_point = labels_patch._get_point_center_of_mass(array)
    assert returned_point == expected_point

    # Check None return if only nan in array
    array = np.full((4, 5), np.nan)
    assert labels_patch._get_point_center_of_mass(array) is None
