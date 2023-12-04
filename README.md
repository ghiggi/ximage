# Welcome to ximage

|                      |                                                |
| -------------------- | ---------------------------------------------- |
| Deployment           | [![PyPI](https://badge.fury.io/py/ximage.svg?style=flat)](https://pypi.org/project/ximage/) [![Conda](https://img.shields.io/conda/vn/conda-forge/ximage.svg?logo=conda-forge&logoColor=white&style=flat)](https://anaconda.org/conda-forge/ximage) |
| Activity             | [![PyPI Downloads](https://img.shields.io/pypi/dm/ximage.svg?label=PyPI%20downloads&style=flat)](https://pypi.org/project/ximage/) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ximage.svg?label=Conda%20downloads&style=flat)](https://anaconda.org/conda-forge/ximage) |
| Python Versions      | [![Python Versions](https://img.shields.io/badge/Python-3.8%20%203.9%20%203.10%20%203.11%20%203.12-blue?style=flat)](https://www.python.org/downloads/) |
| Supported Systems    | [![Linux](https://img.shields.io/github/actions/workflow/status/ghiggi/ximage/.github/workflows/tests.yml?label=Linux&style=flat)](https://github.com/ghiggi/ximage/actions/workflows/tests.yml) [![macOS](https://img.shields.io/github/actions/workflow/status/ghiggi/ximage/.github/workflows/tests.yml?label=macOS&style=flat)](https://github.com/ghiggi/ximage/actions/workflows/tests.yml) [![Windows](https://img.shields.io/github/actions/workflow/status/ghiggi/ximage/.github/workflows/tests_windows.yml?label=Windows&style=flat)](https://github.com/ghiggi/ximage/actions/workflows/tests_windows.yml) |
| Project Status       | [![Project Status](https://www.repostatus.org/badges/latest/active.svg?style=flat)](https://www.repostatus.org/#active) |
| Build Status         | [![Tests](https://github.com/ghiggi/ximage/actions/workflows/tests.yml/badge.svg?style=flat)](https://github.com/ghiggi/ximage/actions/workflows/tests.yml) [![Lint](https://github.com/ghiggi/ximage/actions/workflows/lint.yml/badge.svg?style=flat)](https://github.com/ghiggi/ximage/actions/workflows/lint.yml) [![Docs](https://readthedocs.org/projects/ximage/badge/?version=latest&style=flat)](https://ximage.readthedocs.io/en/latest/) |
| Linting              | [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat)](https://github.com/astral-sh/ruff) [![Codespell](https://img.shields.io/badge/Codespell-enabled-brightgreen?style=flat)](https://github.com/codespell-project/codespell) |
| Code Coverage        | [![Coveralls](https://coveralls.io/repos/github/ghiggi/ximage/badge.svg?branch=main&style=flat)](https://coveralls.io/github/ghiggi/ximage?branch=main) [![Codecov](https://codecov.io/gh/ghiggi/ximage/branch/main/graph/badge.svg?style=flat)](https://codecov.io/gh/ghiggi/ximage) |
| Code Quality         | [![Codefactor](https://www.codefactor.io/repository/github/ghiggi/ximage/badge?style=flat)](https://www.codefactor.io/repository/github/ghiggi/ximage) [![Codebeat](https://codebeat.co/badges/14ff831b-f064-4bdd-a2e2-72ffdf28a35a?style=flat)](https://codebeat.co/projects/github-com-ltelab-ximage-main) [![Codacy](https://app.codacy.com/project/badge/Grade/d823c50a7ad14268bd347b5aba384623?style=flat)](https://app.codacy.com/gh/ghiggi/ximage/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Codescene](https://codescene.io/projects/36773/status-badges/code-health?style=flat)](https://codescene.io/projects/36773) |
| Code Review          | [![pyOpenSci](https://tinyurl.com/XXXX)](#) [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/XXXX/badge?style=flat)](#) |
| License              | [![License](https://img.shields.io/github/license/ghiggi/ximage?style=flat)](https://github.com/ghiggi/ximage/blob/main/LICENSE) |
| Community            | [![Slack](https://img.shields.io/badge/Slack-ximage-green.svg?logo=slack&style=flat)](https://join.slack.com/t/ximageworkspace/shared_invite/zt-25l4mvgo7-cfBdXalzlWGd4Pt7H~FqoA) [![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-green?logo=github&style=flat)](https://github.com/ghiggi/ximage/discussions) |
| Citation             | [![JOSS](http://joss.theoj.org/papers/<DOI>/joss.<DOI>/status.svg?style=flat)](#) [![DOI](https://zenodo.org/badge/664629093.svg?style=flat)](https://zenodo.org/records/8131553) |

 [**Slack**](https://join.slack.com/t/xarray-tools/shared_invite/zt-28f5r0n75-ygNZN5omemhz72NM~WKUHA) | [**Docs**](https://ximage.readthedocs.io/en/latest/)

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

> Ghiggi Gionata & XXXX . ghiggi/ximage. Zenodo. https://doi.org/10.5281/zenodo.8131552

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
