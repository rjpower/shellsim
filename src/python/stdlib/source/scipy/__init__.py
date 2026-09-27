"""shellsim's SciPy.

Submodules load on first access, as in SciPy. ``scipy.linalg``, ``scipy.special`` and
``scipy.stats`` are implemented; the other SciPy submodules raise ``NotImplementedError`` on
import. The behavior targets SciPy 1.18.
"""

__version__ = "1.18.1"

_SUBMODULES = (
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
)


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"scipy.{name}")
    raise AttributeError(f"Module 'scipy' has no attribute '{name}'")


def __dir__():
    return [*globals(), *_SUBMODULES]
