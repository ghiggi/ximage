#!/usr/bin/env python3
"""
Created on Mon Jul 10 14:13:49 2023

@author: ghiggi
"""
import matplotlib.pyplot as plt


def _plot_rectangle(ax, xlim, ylim, edgecolor="red", facecolor="None", **kwargs):
    """Plot rectangles from 2D patch list slices."""
    # Extract the start and stop values from the slice
    x_start, x_stop = xlim
    y_start, y_stop = ylim
    # Calculate the width and height of the rectangle
    width = x_stop - x_start
    height = y_stop - y_start
    # Plot rectangle
    rectangle = plt.Rectangle(
        (x_start, y_start), width, height, edgecolor=edgecolor, facecolor=facecolor, **kwargs
    )
    ax.add_patch(rectangle)
    return ax


def _plot_xr_isel_dict_rectangle(
    ax, xr_obj, label_name, isel_dicts, edgecolor="red", facecolor="None", **kwargs
):
    """Plot xarray 2D isel_dicts rectangles."""
    y, x = list(xr_obj[label_name].dims)
    for isel_dict in isel_dicts:
        xr_subset = xr_obj[label_name].isel(isel_dict)
        _ = _plot_rectangle(
            ax=ax,
            xlim=xr_subset[x].data[[0, -1]],
            ylim=xr_subset[y].data[[0, -1]],
            edgecolor=edgecolor,
            facecolor=facecolor,
            **kwargs,
        )
    return None


def _get_nice_extent_isel_dict(patches_isel_dicts, partitions_isel_dicts, shape_dict):
    # Retrieve name of dimensions
    y, x = list(patches_isel_dicts[0].keys())
    # Retrieve isel_dicts
    isel_dicts = patches_isel_dicts + partitions_isel_dicts
    # Get isel dict covering all isel_dicts
    subset_isel_dicts = {}
    for dim in [y, x]:
        min_start = min([isel_dict[dim].start for isel_dict in isel_dicts])
        max_stop = max([isel_dict[dim].stop for isel_dict in isel_dicts])
        # Extend a bit
        min_start = max(min_start - 2, 0)
        max_stop = min(max_stop + 2, shape_dict[dim])
        subset_isel_dicts[dim] = slice(min_start, max_stop)
    return subset_isel_dicts


def plot_label_patch_extraction_areas(
    xr_obj, label_name, patches_isel_dicts, partitions_isel_dicts, **kwargs
):
    """Plot for debugging label patch extraction."""
    from ximage.labels.plot_labels import plot_labels

    # Get isel dict covering all isel_dicts
    shape_dict = {dim: xr_obj[dim].shape[0] for dim in xr_obj[label_name].dims}
    subset_isel_dicts = _get_nice_extent_isel_dict(
        patches_isel_dicts, partitions_isel_dicts, shape_dict=shape_dict
    )
    # Subset the label array to plot
    label_subset = xr_obj[label_name].isel(subset_isel_dicts)
    # Create figure
    fig, ax = plt.subplots()
    # Plot labels
    p = plot_labels(label_subset, ax=ax)
    p.axes.set_aspect("equal")
    # Plot partitions rectangles
    _ = _plot_xr_isel_dict_rectangle(
        ax=ax,
        xr_obj=xr_obj,
        label_name=label_name,
        isel_dicts=partitions_isel_dicts,
        edgecolor="black",
        facecolor="None",
        **kwargs,
    )
    # Plot patches rectangles
    _ = _plot_xr_isel_dict_rectangle(
        ax=ax,
        xr_obj=xr_obj,
        label_name=label_name,
        isel_dicts=patches_isel_dicts,
        edgecolor="red",
        facecolor="None",
        **kwargs,
    )
    return fig
