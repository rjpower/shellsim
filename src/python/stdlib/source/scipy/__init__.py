"""shellsim's ``scipy``: a small subset of SciPy on top of shellsim's NumPy.

Subpackages load on first attribute access through a module ``__getattr__`` (PEP 562), so
``import scipy`` stays cheap and ``scipy.stats`` works without importing it first. Importing a
submodule binds it on this package, so the hook runs once per subpackage. Subpackages shellsim
does not provide are missing modules: importing one raises ``ModuleNotFoundError``.
"""

import importlib

__all__ = ["integrate", "interpolate", "linalg", "spatial", "special", "stats"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module 'scipy' has no attribute {name!r}")
    return importlib.import_module(f"scipy.{name}")
