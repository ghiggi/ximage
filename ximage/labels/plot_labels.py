#!/usr/bin/env python3
"""
Created on Tue Jul 11 10:21:28 2023

@author: ghiggi
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ximage.labels.labels import get_label_indices, redefine_label_array


def get_label_colorbar_settings(label_indices, cmap="Paired"):
    """Return plot and cbar kwargs to plot properly a label array."""
    # Cast to int the label_indices
    label_indices = label_indices.astype(int)
    # Compute number of required colors
    n_labels = len(label_indices)

    # Get colormap if string
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Extract colors
    color_list = [cmap(i) for i in range(cmap.N)]

    # Create the new colormap
    cmap_new = mpl.colors.LinearSegmentedColormap.from_list("Label Classes", color_list, n_labels)

    # Define the bins and normalize
    bounds = np.linspace(1, n_labels + 1, n_labels + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap_new.N)

    # Define the plot kwargs
    plot_kwargs = {}
    plot_kwargs["cmap"] = cmap_new
    plot_kwargs["norm"] = norm

    # Define colorbar kwargs
    ticks = bounds[:-1] + 0.5
    ticklabels = label_indices
    assert len(ticks) == len(ticklabels)
    cbar_kwargs = {}
    cbar_kwargs["label"] = "Label IDs"
    cbar_kwargs["ticks"] = ticks
    cbar_kwargs["ticklabels"] = ticklabels
    return plot_kwargs, cbar_kwargs


def plot_labels(
    dataarray,
    x=None,
    y=None,
    ax=None,
    max_n_labels=50,
    add_colorbar=True,
    cmap="Paired",
    **plot_kwargs,
):
    """Plot labels.

    The maximum allowed number of labels to plot is 'max_n_labels'.
    """
    dataarray = dataarray.compute()
    label_indices = get_label_indices(dataarray)
    n_labels = len(label_indices)
    if add_colorbar and n_labels > max_n_labels:
        msg = f"""The array currently contains {n_labels} labels and 'max_n_labels'
            is set to {max_n_labels}. The colorbar is not displayed!"""
        print(msg)
        add_colorbar = False
    dataarray = redefine_label_array(dataarray, label_indices=label_indices)
    # Replace 0 with nan
    dataarray = dataarray.where(dataarray > 0)
    # Define appropriate colormap
    plot_kwargs, cbar_kwargs = get_label_colorbar_settings(label_indices, cmap="Paired")
    # Plot image
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    if not add_colorbar:
        cbar_kwargs = {}

    p = dataarray.plot.imshow(
        x=x,
        y=y,
        ax=ax,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )
    plt.title(dataarray.name)
    if add_colorbar and ticklabels is not None:
        p.colorbar.ax.set_yticklabels(ticklabels)
    return p
