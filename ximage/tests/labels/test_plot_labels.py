from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ximage.labels.plot_labels import plot_labels


def test_plot_labels():
    """Test plot_labels"""

    n_labels = 10
    shape = (10, 10)
    array = np.random.randint(0, n_labels, size=shape)
    array = array.astype(float)
    array[array == 0] = np.nan

    x_dim = "x"
    y_dim = "y"
    dataarray = xr.DataArray(array, dims=[y_dim, x_dim])

    # Default arguments
    p = plot_labels(dataarray)
    assert isinstance(p, AxesImage)

    # Passing arguments to xarray imshow
    ax = plt.gca()
    p = plot_labels(dataarray, x=x_dim, y=y_dim, ax=ax)
    assert isinstance(p, AxesImage)

    # Cap max_n_labels to hide colorbar
    p = plot_labels(dataarray, max_n_labels=1)
    assert isinstance(p, AxesImage)
