# Welcome to ximage
[![DOI](https://zenodo.org/badge/286664485.svg)](https://zenodo.org/badge/latestdoi/286664485)
[![PyPI version](https://badge.fury.io/py/ximage.svg)](https://badge.fury.io/py/ximage)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ximage.svg)](https://anaconda.org/conda-forge/ximage)
[![Tests](https://github.com/ghiggi/ximage/actions/workflows/tests.yml/badge.svg)](https://github.com/ghiggi/ximage/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/ghiggi/ximage/badge.svg?branch=main)](https://coveralls.io/github/ghiggi/ximage?branch=main)
[![Documentation Status](https://readthedocs.org/projects/ximage/badge/?version=latest)](https://ximage.readthedocs.io/projects/ximage/en/stable/?badge=stable)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License](https://img.shields.io/github/license/ghiggi/ximage)](https://github.com/ghiggi/ximage/blob/master/LICENSE)

The ximage is still in development. Feel free to try it out and to report issues or to suggest changes.

## Quick start

ximage provides an easy-to-use interface to manipulate image, videos and n-dimensional arrays
with classical image processing techniques.

Look at the [Tutorials][tutorial_link] to have an overview of the software !

## Installation

### pip

ximage can be installed via [pip][pip_link] on Linux, Mac, and Windows.
On Windows you can install [WinPython][winpy_link] to get Python and pip
running.
Then, install the ximage package by typing the following command in the command terminal:

    pip install ximage

To install the latest development version via pip, see the
[documentation][doc_install_link].

### conda [NOT YET AVAILABLE]

ximage can be installed via [conda][conda_link] on Linux, Mac, and Windows.
Install the package by typing the following command in a command terminal:

    conda install ximage

In case conda forge is not set up for your system yet, see the easy to follow
instructions on [conda forge][conda_forge_link].


## Documentation for ximage

You can find the documentation under [ximage.readthedocs.io][doc_link]

### Tutorials and Examples

The documentation also includes some [tutorials][tutorial_link], showing the most important use cases of ximage.
These tutorial are also available as Jupyter Notebooks and in Google Colab:

- 1. Introduction to image labeling [[Notebook][tut3_label_link]][[Colab][colab3_label_link]]
- 2. Introduction to label patch extraction [[Notebook][tut3_label_link]][[Colab][colab3_label_link]]
- 3. Introduction to image patch extraction [[Notebook][tut3_patch_link]][[Colab][colab3_patch_link]]


## Citation

If you are using ximage in your publication please cite our paper:

TODO: GMD

You can cite the Zenodo code publication of ximage by:

> Ghiggi Gionata & XXXX . ghiggi/ximage. Zenodo. https://doi.org/10.5281/zenodo.7753488

If you want to cite a specific version, have a look at the [Zenodo site](https://doi.org/10.5281/zenodo.7753488).

## Requirements:

- [xarray](https://docs.xarray.dev/en/stable/)
- [dask](https://www.dask.org/)
- [dask_image](https://image.dask.org/en/latest/)
- [skimage](https://scikit-image.org/)


## License

The content of this repository is released under the terms of the [MIT](LICENSE) license.


[pip_link]: https://pypi.org/project/gstools
[conda_link]: https://docs.conda.io/en/latest/miniconda.html
[conda_forge_link]: https://github.com/conda-forge/ximage-feedstock#installing-ximage
[conda_pip]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages
[pipiflag]: https://pip-python3.readthedocs.io/en/latest/reference/pip_install.html?highlight=i#cmdoption-i
[winpy_link]: https://winpython.github.io/

[tutorial_link]: https://github.com/ghiggi/ximage/tree/master#tutorials-and-examples

[tut3_label_link]: https://github.com/ghiggi/ximage/tree/master/tutorials
[colab3_label_link]: https://github.com/ghiggi/ximage/tree/master/tutorials

[tut3_patch_link]: https://github.com/ghiggi/ximage/tree/master/tutorials
[colab3_patch_link]: https://github.com/ghiggi/ximage/tree/master/tutorials
