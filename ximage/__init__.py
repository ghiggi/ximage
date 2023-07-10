#!/usr/bin/env python3
"""
Created on Mon Aug 15 00:17:07 2022

@author: ghiggi
"""
import ximage.accessor  # noqa
from ximage.labels.labels import label
from ximage.patch.labels_patch import get_patches_from_labels as label_patches
from importlib.metadata import PackageNotFoundError, version


__all__ = ["label", "label_patches"]

# Get version
try:
    __version__ = version("ximage")
except PackageNotFoundError:
    # package is not installed
    pass
