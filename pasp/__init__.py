"""
.. include:: ../README.md
"""

from .grammar import parse
from exact import exact, count
from ground import ground
from .program import Program
from sample import sample
from .wlearn import learn
import approx

import numpy as np

try:
    from importlib.metadata import version
    __version__ = version("pasp-plp")
except Exception:
    __version__ = "unknown"

# Pre-compile CUDA kernel if GPU available (cached on disk after first run).
try:
    from .gpu_optimize import warmup as _gpu_warmup
    _gpu_warmup()
except Exception:
    pass
