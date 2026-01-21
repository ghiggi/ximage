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
"""ximage xarray-accessor methods."""
import numpy as np
import xarray as xr


class XImage_Base_Accessor:
    """XImage Base Accessor for xarray objects."""

    def __init__(self, xarray_obj):
        """Create a new XImage_Base_Accessor object."""
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError("The 'ximage' accessor is available only for xr.Dataset and xr.DataArray.")
        self._obj = xarray_obj

    def label(
        self,
        *,
        variable=None,
        core_dims=None,
        min_value_threshold=0.1,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint=None,
        sort_by="area",
        sort_decreasing=True,
        labeled_comprehension_kwargs=None,
        label_name="label",
    ):
        """Label the xarray object."""
        from ximage.labels.labels import label

        if labeled_comprehension_kwargs is None:
            labeled_comprehension_kwargs = {}
        return label(
            self._obj,
            variable=variable,
            core_dims=core_dims,
            label_name=label_name,
            # Labels options
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint=footprint,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            labeled_comprehension_kwargs=labeled_comprehension_kwargs,
        )

    def label_patches(
        self,
        *,
        patch_size,
        variable=None,
        label_name="label",
        # Output options
        n_patches=None,
        n_labels=None,
        labels_id=None,
        highlight_label_id=True,
        # Label Patch Extraction Options
        centered_on="max",
        padding=0,
        n_patches_per_label=None,
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
        """Extract patches around labels of the xarray DataArray ``label_name``."""
        from ximage.patch.labels_patch import get_patches_from_labels

        return get_patches_from_labels(
            self._obj,
            label_name=label_name,
            patch_size=patch_size,
            variable=variable,
            # Output options
            n_patches=n_patches,
            n_labels=n_labels,
            labels_id=labels_id,
            highlight_label_id=highlight_label_id,
            # Patch extraction Options
            padding=padding,
            centered_on=centered_on,
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
            # Other Options
            verbose=verbose,
            debug=debug,
        )

    def label_patches_isel_dicts(
        self,
        *,
        label_name,
        patch_size,
        variable=None,
        # Output options
        n_patches=None,
        n_labels=None,
        labels_id=None,
        # Label Patch Extraction Settings
        centered_on="max",
        padding=0,
        n_patches_per_label=None,
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
        """Return isel-dictionaries to extract patches around labels."""
        from ximage.patch.labels_patch import get_patches_isel_dict_from_labels

        return get_patches_isel_dict_from_labels(
            self._obj,
            label_name=label_name,
            patch_size=patch_size,
            variable=variable,
            # Output options
            n_patches=n_patches,
            n_labels=n_labels,
            labels_id=labels_id,
            # Patch extraction Options
            padding=padding,
            centered_on=centered_on,
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
            # Other Options
            verbose=verbose,
            debug=debug,
        )


@xr.register_dataset_accessor("ximage")
class XImage_Dataset_Accessor(XImage_Base_Accessor):
    """XImage Dataset Accessor for xarray objects."""

    def __init__(self, xarray_obj):
        """Initialize a XImage_Dataset_Accessor object."""
        super().__init__(xarray_obj)


@xr.register_dataarray_accessor("ximage")
class XImage_DataArray_Accessor(XImage_Base_Accessor):
    """XImage DataArray Accessor for xarray objects."""

    def __init__(self, xarray_obj):
        """Initialize a XImage_DataArray_Accessor object."""
        super().__init__(xarray_obj)

    def plot_labels(
        self,
        *,
        x=None,
        y=None,
        ax=None,
        max_n_labels=50,
        add_colorbar=True,
        cmap="Paired",
        use_imshow=False,
        **plot_kwargs,
    ):
        """Plot the labels on the xarray object."""
        from ximage.labels.plot_labels import plot_labels

        return plot_labels(
            self._obj,
            x=x,
            y=y,
            ax=ax,
            max_n_labels=max_n_labels,
            add_colorbar=add_colorbar,
            cmap=cmap,
            use_imshow=use_imshow,
            **plot_kwargs,
        )
