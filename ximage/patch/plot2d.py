#!/usr/bin/env python3
"""
Created on Mon Jul 10 14:13:49 2023

@author: ghiggi
"""
import matplotlib.pyplot as plt


def plot_rectangle_from_list_slices(ax, list_slices, edgecolor="red", facecolor="None", **kwargs):
    """Plot rectangles from 2D patch list slices."""
    if len(list_slices) != 2:
        raise ValueError("Required 2 slices.")
    # Extract the start and stop values from the slice
    y_start, y_stop = (list_slices[0].start, list_slices[0].stop)
    x_start, x_stop = (list_slices[1].start, list_slices[1].stop)
    # Calculate the width and height of the rectangle
    width = x_stop - x_start
    height = y_stop - y_start
    # Plot rectangle
    rectangle = plt.Rectangle(
        (x_start, y_start), width, height, edgecolor=edgecolor, facecolor=facecolor, **kwargs
    )
    ax.add_patch(rectangle)
    return ax


def plot_2d_label_partitions_boundaries(
    partitions_list_slices, label_arr, edgecolor="red", facecolor="None", **kwargs
):
    """Plot partitions from 2D list slices."""
    # Define plot limits
    xmin = min([patch_list_slices[1].start for patch_list_slices in partitions_list_slices])
    xmax = max([patch_list_slices[1].stop for patch_list_slices in partitions_list_slices])
    ymin = min([patch_list_slices[0].start for patch_list_slices in partitions_list_slices])
    ymax = max([patch_list_slices[0].stop for patch_list_slices in partitions_list_slices])

    # Plot patches boundaries
    fig, ax = plt.subplots()
    ax.imshow(label_arr, origin="upper")
    for partition_list_slices in partitions_list_slices:
        _ = plot_rectangle_from_list_slices(
            ax=ax,
            list_slices=partition_list_slices,
            edgecolor=edgecolor,
            facecolor=facecolor,
            **kwargs,
        )
    # Set plot limits
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymax + 5, ymin - 5)
    return fig


def add_label_patches_boundaries(
    fig, patches_list_slices, edgecolor="red", facecolor="None", **kwargs
):

    # Retrieve axis
    ax = fig.axes[0]

    # Define patches limits
    xmin = min([patch_list_slices[1].start for patch_list_slices in patches_list_slices])
    xmax = max([patch_list_slices[1].stop for patch_list_slices in patches_list_slices])
    ymin = min([patch_list_slices[0].start for patch_list_slices in patches_list_slices])
    ymax = max([patch_list_slices[0].stop for patch_list_slices in patches_list_slices])

    # Get current plot axis limits
    plot_xmin, plot_xmax = ax.get_xlim()
    plot_ymin, plot_ymax = ax.get_ylim()

    # Define final plot axis limits
    xmin = min(xmin, plot_xmin)
    xmax = max(xmax, plot_xmax)
    ymin = min(ymin, plot_ymin)
    ymax = max(ymax, plot_ymax)

    # Plot patch boundaries
    for patch_list_slices in patches_list_slices:
        _ = plot_rectangle_from_list_slices(
            ax=ax, list_slices=patch_list_slices, edgecolor=edgecolor, facecolor=facecolor, **kwargs
        )
    # Set plot limits
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymax + 5, ymin - 5)

    return fig


def plot_2d_label_patches_boundaries(patches_list_slices, label_arr):
    """Plot patches from  from 2D list slices."""
    # Define plot limits
    xmin = min([patch_list_slices[1].start for patch_list_slices in patches_list_slices])
    xmax = max([patch_list_slices[1].stop for patch_list_slices in patches_list_slices])
    ymin = min([patch_list_slices[0].start for patch_list_slices in patches_list_slices])
    ymax = max([patch_list_slices[0].stop for patch_list_slices in patches_list_slices])

    # Plot patches boundaries
    fig, ax = plt.subplots()
    ax.imshow(label_arr, origin="upper")
    for patch_list_slices in patches_list_slices:
        plot_rectangle_from_list_slices(ax, patch_list_slices)

    # Set plot limits
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymax + 5, ymin - 5)

    # Show plot
    plt.show()

    # Return figure
    return fig
