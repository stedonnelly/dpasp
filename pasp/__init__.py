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
# Runs in a background thread to avoid blocking import on cold JIT compilation.
try:
    from .gpu_optimize import warmup as _gpu_warmup
    import threading as _threading
    _threading.Thread(target=_gpu_warmup, daemon=True).start()
except Exception:
    pass
