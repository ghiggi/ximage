import inspect

from ximage.accessor.methods import XImage_Base_Accessor, XImage_DataArray_Accessor
from ximage.labels.labels import label
from ximage.patch.labels_patch import get_patches_from_labels
from ximage.patch.labels_patch import get_patches_isel_dict_from_labels
from ximage.labels.plot_labels import plot_labels


# Utils functions ##############################################################


def get_default_arguments_dict(function):
    """Get default arguments of a function as a dictionary"""

    signature = inspect.signature(function)
    default_arguments = {}

    for key, value in signature.parameters.items():
        if value.default is not inspect.Parameter.empty:
            default_arguments[key] = value.default

    return default_arguments


def compare_default_arguments(base_function, reference_function, changed_default_arguments={}):
    """Check that default arguments of base_function and reference_function are the same"""

    base_default_arguments = get_default_arguments_dict(base_function)
    reference_default_arguments = get_default_arguments_dict(reference_function)

    modified_default_arguments = {**base_default_arguments, **changed_default_arguments}

    assert modified_default_arguments == reference_default_arguments


# Tests for accessors' methods default arguments ###############################


def test_base_accessor_label():
    """Check default arguments of XImage_Base_Accessor.label"""

    changed_default_arguments = {
        "min_value_threshold": 0.1,
    }
    compare_default_arguments(label, XImage_Base_Accessor.label, changed_default_arguments)


def test_base_accessor_label_patches():
    """Check default arguments of XImage_Base_Accessor.label_patches"""

    changed_default_arguments = {
        "label_name": "label",
    }
    compare_default_arguments(
        get_patches_from_labels, XImage_Base_Accessor.label_patches, changed_default_arguments
    )


def test_base_accessor_label_patches_isel_dict():
    """Check default arguments of XImage_Base_Accessor.label_patches_isel_dict"""

    compare_default_arguments(
        get_patches_isel_dict_from_labels, XImage_Base_Accessor.label_patches_isel_dicts
    )


def test_dataarray_accessor_plot_labels():
    """Check default arguments of XImage_DataArray_Accessor.plot_labels"""

    compare_default_arguments(plot_labels, XImage_DataArray_Accessor.plot_labels)
