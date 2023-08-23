import numpy as np
import pytest
import xarray as xr

from pytest import SaneEqualityArray

from ximage.patch import labels_patch

# Utils functions ##############################################################


def check_slices_and_patches(
    args, kwargs, variable_array, variable_name, labels_array, label_name, expected_slices_dict
):
    """Utils function for testing get_patches_from_labels and get_patches_isel_dict_from_labels.

    Calls the two methods with "args" and "kwargs", checks that the returned slices are the same as the expected ones,
    and checks that the patches have the same sliced variable and labels arrays.
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

    variable_array = np.reshape(np.arange(shape[0] * shape[1]), shape)

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
        "centered_on": "centroid",
        "partitioning_method": "tiling",
    }
    expected_slices = {
        1: [
            {"x": slice(1, 4), "y": slice(1, 4)},
            {"x": slice(4, 7), "y": slice(4, 7)},
            {
                "x": slice(5, 8),
                "y": slice(5, 8),
            },  # From (4, 7)(6, 9) and (6, 9)(4, 7) that got recentered
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


# Tests for internal functions #################################################


def test_check_label_arr():
    """Test _check_label_arr"""
