"""shellsim's ``scipy``.

Real SciPy loads each subpackage lazily on first attribute access instead of importing all of
them up front. shellsim reproduces that with a module ``__getattr__`` (PEP 562): touching
``scipy.stats`` imports the ``scipy.stats`` submodule the first time and caches it as a module
attribute afterwards, exactly as ``import scipy.stats`` would. Only ``special``, ``stats``,
``linalg`` and ``spatial`` do anything; the rest resolve to ``scipy._unsupported``, whose import
always raises ``NotImplementedError``.
"""

import importlib

__all__ = [
    "cluster",
    "constants",
    "datasets",
    "differentiate",
    "fft",
    "fftpack",
    "integrate",
    "interpolate",
    "io",
    "linalg",
    "ndimage",
    "odr",
    "optimize",
    "signal",
    "sparse",
    "spatial",
    "special",
    "stats",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"scipy.{name}")
    globals()[name] = module
    return module
