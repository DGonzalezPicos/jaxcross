from .cross_correlation import CCF
from .template import Template

import sys

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

try:
    __version__ = metadata.version(__package__ or __name__)
except:
    __version__ = "dev"
