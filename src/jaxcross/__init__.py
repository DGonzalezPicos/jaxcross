from .cross_correlation import CCF, KpV
from .template import Template
from .crires import CRIRES
from .planet import Planet
from .align import Align
from .interpolate import InterpolatedUnivariateSpline
import sys

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

try:
    __version__ = metadata.version(__package__ or __name__)
except:
    __version__ = "dev"
