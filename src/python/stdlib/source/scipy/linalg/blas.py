"""shellsim's ``scipy.linalg.blas``: f2py-style wrappers over BLAS routines.

Only the real single- and double-precision ``nrm2`` (Euclidean norm) is modeled, through the
native ``_scipy_linalg.nrm2`` kernel. Every other BLAS routine name, and the complex (`c`/`z`)
forms of the modeled one, raises ``NotImplementedError`` on attribute access or through
:func:`get_blas_funcs`, matching the unsupported frontier ``docs/scipy.md`` documents.
"""

import numpy as np
from _scipy_linalg import nrm2 as _nrm2

__all__ = ["get_blas_funcs", "find_best_blas_type"]

_TYPECODES = {
    np.dtype(np.float32): "s",
    np.dtype(np.float64): "d",
    np.dtype(np.complex64): "c",
    np.dtype(np.complex128): "z",
}
_RANK = {"s": 0, "d": 1, "c": 2, "z": 3}
_SUPPORTED = {"snrm2", "dnrm2"}


def _resolve_dtype(dtype):
    """The BLAS-modeled dtype nearest `dtype`: itself if it is one of the four BLAS types,
    else `complex128` for another complex type or `float64` for anything else (real SciPy
    resolves through the same four kinds, widening integers and `float16` to `float64`)."""
    dtype = np.dtype(dtype)
    if dtype in _TYPECODES:
        return dtype
    if np.issubdtype(dtype, np.complexfloating):
        return np.dtype(np.complex128)
    return np.dtype(np.float64)


def find_best_blas_type(arrays=(), dtype=None):
    """The BLAS type prefix, resolved dtype, and whether Fortran order is preferred.

    Scans `arrays` (converted with :func:`numpy.asarray`) for the highest-priority dtype among
    `float32 < float64 < complex64 < complex128`, starting from `dtype` if given. Fortran order
    is preferred if any array is Fortran-contiguous (as a 0-d or 1-D array always is).
    """
    best = _resolve_dtype(dtype) if dtype is not None else np.dtype(np.float32)
    best_rank = _RANK[_TYPECODES[best]] if dtype is not None else -1
    prefer_fortran = False
    for value in arrays:
        array = np.asarray(value)
        candidate = _resolve_dtype(array.dtype)
        rank = _RANK[_TYPECODES[candidate]]
        if rank > best_rank:
            best_rank = rank
            best = candidate
        if array.flags.f_contiguous:
            prefer_fortran = True
    return _TYPECODES[best], best, prefer_fortran


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
        return _FortranFunction(name, prefix, dtype, "fblas", _nrm2)
    if len(name) > 1 and name[0] in "sdcz":
        raise NotImplementedError(f"BLAS routine {name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module 'scipy.linalg.blas' has no attribute '{name}'")


def get_blas_funcs(names, arrays=(), dtype=None):
    """Real single/double BLAS wrappers for `names`, resolved from `arrays`' dtypes.

    `names` is either one routine's short name (returning one wrapper) or a sequence of them
    (returning a list, in order).
    """
    prefix, resolved, _ = find_best_blas_type(arrays, dtype)
    single = isinstance(names, str)
    requested = [names] if single else list(names)
    functions = [__getattr__(f"{prefix}{name}") for name in requested]
    return functions[0] if single else functions
