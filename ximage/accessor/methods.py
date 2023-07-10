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
        label_name="label",
    ):
        from ximage.label import label

        xr_obj = label(
            self._obj,
            variable=variable,
            # Labels options
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint=footprint,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
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
    ):
        from ximage.label import get_patches_from_labels

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
        )
        return gen


@xr.register_dataset_accessor("ximage")
class XImage_Dataset_Accessor(XImage_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)


@xr.register_dataarray_accessor("ximage")
class XImage_DataArray_Accessor(XImage_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
