#!/usr/bin/env python3
"""
Created on Mon Jul 10 13:36:29 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr


class XImage_Base_Accessor:
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError(
                "The 'ximage' accessor is available only for xr.Dataset and xr.DataArray."
            )
        self._obj = xarray_obj

    def label(
        self,
        variable=None,
        min_value_threshold=0.1,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint=None,
        sort_by="area",
        sort_decreasing=True,
        labeled_comprehension_kwargs={},
        label_name="label",
    ):
        from ximage.labels.labels import label

        xr_obj = label(
            self._obj,
            variable=variable,
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
        return xr_obj

    def label_patches(
        self,
        patch_size,
        variable=None,
        label_name="label",
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
        from ximage.patch.labels_patch import get_patches_from_labels

        gen = get_patches_from_labels(
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
        return gen

    def label_patches_isel_dicts(
        self,
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
        from ximage.patch.labels_patch import get_patches_isel_dict_from_labels

        isel_dicts = get_patches_isel_dict_from_labels(
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
        return isel_dicts


@xr.register_dataset_accessor("ximage")
class XImage_Dataset_Accessor(XImage_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)


@xr.register_dataarray_accessor("ximage")
class XImage_DataArray_Accessor(XImage_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def plot_labels(
        self,
        x=None,
        y=None,
        ax=None,
        max_n_labels=50,
        add_colorbar=True,
        cmap="Paired",
        **plot_kwargs,
    ):
        from ximage.labels.plot_labels import plot_labels

        return plot_labels(
            self._obj,
            x=x,
            y=y,
            ax=ax,
            max_n_labels=max_n_labels,
            add_colorbar=add_colorbar,
            cmap=cmap,
            **plot_kwargs,
        )
