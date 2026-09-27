"""shellsim's ``scipy.linalg.lapack``: f2py-style wrappers over LAPACK routines.

The real single- and double-precision forms of ``getrf``, ``getrs``, ``gecon``, ``getri``,
``trtrs``, ``trtri``, ``potrf``, ``potrs``, ``potri``, ``gtsv``, ``gbsv`` and ``lange`` are
modeled, each calling the matching ``_scipy_linalg`` native routine directly with f2py's own
argument names, defaults, argument checks, and Fortran-ordered outputs (see
``src/python/stdlib/scipy/linalg/lapack.rs``). Every other LAPACK routine name, and the complex
(`c`/`z`) forms of the modeled ones, raises ``NotImplementedError`` on attribute access or
through :func:`get_lapack_funcs`, matching the unsupported frontier ``docs/scipy.md`` documents.
"""

import numpy as np
import _scipy_linalg
from scipy.linalg.blas import find_best_blas_type as find_best_lapack_type

__all__ = ["get_lapack_funcs", "find_best_lapack_type"]

_ROUTINES = (
    "getrf",
    "getrs",
    "gecon",
    "getri",
    "trtrs",
    "trtri",
    "potrf",
    "potrs",
    "potri",
    "gtsv",
    "gbsv",
    "lange",
)
_SUPPORTED = {f"{prefix}{routine}" for prefix in "sd" for routine in _ROUTINES}


class _FortranFunction:
    """A callable that looks and prints like an f2py-wrapped Fortran routine."""

    def __init__(self, label, typecode, dtype, module_name, call):
        self.__name__ = f"function {label}"
        self.typecode = typecode
        self.prefix = typecode
        self.dtype = dtype
        self.module_name = module_name
        self._label = label
        self._call = call

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __repr__(self):
        return f"<fortran function {self._label}>"


def __getattr__(name):
    if name in _SUPPORTED:
        prefix = name[0]
        dtype = np.dtype(np.float32 if prefix == "s" else np.float64)
        native = getattr(_scipy_linalg, name)
        return _FortranFunction(name, prefix, dtype, "flapack", native)
    if len(name) > 1 and name[0] in "sdcz":
        raise NotImplementedError(f"LAPACK routine {name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module 'scipy.linalg.lapack' has no attribute '{name}'")


def get_lapack_funcs(names, arrays=(), dtype=None):
    """Real single/double LAPACK wrappers for `names`, resolved from `arrays`' dtypes.

    `names` is either one routine's short name (returning one wrapper) or a sequence of them
    (returning a list, in order).
    """
    prefix, resolved, _ = find_best_lapack_type(arrays, dtype)
    single = isinstance(names, str)
    requested = [names] if single else list(names)
    functions = [__getattr__(f"{prefix}{name}") for name in requested]
    return functions[0] if single else functions
