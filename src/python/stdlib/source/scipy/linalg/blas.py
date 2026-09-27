"""shellsim's ``scipy.linalg.blas``, following SciPy 1.18's ``scipy/linalg/blas.py``.

SciPy exposes its f2py-compiled BLAS wrappers here, one per precision prefix (``s``, ``d``,
``c``, ``z``). shellsim provides ``snrm2`` and ``dnrm2``, computed by the native
``_scipy_linalg`` module. The other routines, and every complex routine, raise
``NotImplementedError`` when fetched through ``get_blas_funcs`` or read as module attributes.

Each routine is an ``_Routine`` object that stands in for an f2py ``fortran`` object: it takes
f2py's keyword arguments, casts array arguments to its precision, and carries the ``dtype``,
``typecode``, ``prefix``, ``module_name`` and ``int_dtype`` attributes that ``_get_funcs`` sets.
Results are C-ordered where f2py returns Fortran-ordered arrays with the same values.
"""

import numpy as np
import _scipy_linalg as _native

__all__ = ["find_best_blas_type", "get_blas_funcs"]

HAS_LP64 = True
HAS_ILP64 = False

_PREFIX_DTYPES = {"s": np.dtype("float32"), "d": np.dtype("float64")}


class _Routine:
    """One precision of a BLAS or LAPACK routine, called with f2py's arguments."""

    def __init__(self, name, function, module_name):
        prefix = name[0]
        # f2py names its routine objects "function dnrm2".
        self.__name__ = f"function {name}"
        self._function = function
        self.module_name = module_name
        self.typecode = prefix
        self.prefix = prefix
        self.dtype = _PREFIX_DTYPES[prefix]
        self.int_dtype = np.dtype(np.intc)

    def __call__(self, *args, **kwargs):
        return self._function(self.dtype, *args, **kwargs)

    def __repr__(self):
        return f"<fortran {self.__name__}>"


def _as(value, dtype):
    """An f2py array argument: ``value`` as an array of ``dtype``."""
    return np.asarray(value, dtype=dtype)


def _nrm2(dtype, x, n=None, offx=0, incx=1):
    if offx < 0:
        raise ValueError("offx must be nonnegative")
    if incx <= 0:
        raise NotImplementedError("nrm2 with incx <= 0 is not supported by shellsim's SciPy")
    x = _as(x, dtype)
    if n is None:
        # f2py's default, which can leave out the last element of a strided vector.
        n = (len(x) - offx) // incx
    return _native.nrm2(x[offx::incx][:n])


def _define(functions, module_name):
    """The ``s`` and ``d`` routines for each of ``functions``, by name."""
    routines = {}
    for name, function in functions.items():
        for prefix in "sd":
            routines[prefix + name] = _Routine(prefix + name, function, module_name)
    return routines


_ROUTINES = _define({"nrm2": _nrm2}, "fblas")


def _unsupported(kind, name):
    return NotImplementedError(f"{kind} routine {name} is not supported by shellsim's SciPy")


def __getattr__(name):
    # The routines are served from here because shellsim has no `globals()` to define them with.
    if name in _ROUTINES:
        return _ROUTINES[name]
    if len(name) > 1 and name[0] in "sdcz":
        raise _unsupported("BLAS", name)
    raise AttributeError(f"module 'scipy.linalg.blas' has no attribute '{name}'")


_type_score = {x: 1 for x in "?bBhHef"}
_type_score.update({x: 2 for x in "iIlLqQd"})
_type_score.update({"F": 3, "D": 4, "g": 2, "G": 4})

_type_conv = {
    1: ("s", np.dtype("float32")),
    2: ("d", np.dtype("float64")),
    3: ("c", np.dtype("complex64")),
    4: ("z", np.dtype("complex128")),
}

_blas_alias = {
    "cnrm2": "scnrm2",
    "znrm2": "dznrm2",
    "cdot": "cdotc",
    "zdot": "zdotc",
    "cger": "cgerc",
    "zger": "zgerc",
    "sdotc": "sdot",
    "sdotu": "sdot",
    "ddotc": "ddot",
    "ddotu": "ddot",
}


def find_best_blas_type(arrays=(), dtype=None):
    """The BLAS prefix, dtype and preferred memory order for ``arrays`` or ``dtype``.

    ``float32`` and smaller types give ``'s'``, ``float64`` and the integers ``'d'``, and the
    complex types ``'c'`` and ``'z'``; a mix of ``float64`` and ``complex64`` gives ``'z'``.
    """
    dtype = np.dtype(dtype)
    max_score = _type_score.get(dtype.char, 5)
    prefer_fortran = False
    if arrays:
        if len(arrays) == 1:
            max_score = _type_score.get(arrays[0].dtype.char, 5)
            prefer_fortran = arrays[0].flags["FORTRAN"]
        else:
            scores = [_type_score.get(x.dtype.char, 5) for x in arrays]
            max_score = max(scores)
            ind_max_score = scores.index(max_score)
            if max_score == 3 and (2 in scores):
                max_score = 4
            if arrays[ind_max_score].flags["FORTRAN"]:
                prefer_fortran = True
    prefix, dtype = _type_conv.get(max_score, ("d", np.dtype("float64")))
    return prefix, dtype, prefer_fortran


def _get_funcs(names, arrays, dtype, lib_name, fmodule, fmodule_name, alias, ilp64="preferred"):
    """The routines ``names`` of ``fmodule`` for the precision ``arrays`` or ``dtype`` need."""
    funcs = []
    unpack = False
    dtype = np.dtype(dtype)
    if isinstance(names, str):
        names = (names,)
        unpack = True
    prefix, dtype, _ = find_best_blas_type(arrays, dtype)
    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = fmodule.get(func_name)
        if func is None:
            raise _unsupported(lib_name, func_name)
        funcs.append(func)
    return funcs[0] if unpack else funcs


def _resolve_ilp64(kind, ilp64):
    if isinstance(ilp64, str):
        if ilp64 == "preferred":
            return HAS_ILP64
        raise ValueError(f"Invalid value for {ilp64 = }.")
    if ilp64:
        raise RuntimeError(
            f"{kind} ILP64 routine requested, but Scipy compiled only with 32-bit {kind}"
        )
    return False


def get_blas_funcs(names, arrays=(), dtype=None, ilp64="preferred"):
    """BLAS routines by name, in the precision that ``arrays`` or ``dtype`` call for.

    A single name gives a single routine; a sequence of names gives a list.

    >>> nrm2 = get_blas_funcs("nrm2", (np.array([3.0, 4.0]),))
    >>> nrm2.typecode, nrm2(np.array([3.0, 4.0]))
    ('d', 5.0)
    """
    _resolve_ilp64("BLAS", ilp64)
    return _get_funcs(names, arrays, dtype, "BLAS", _ROUTINES, "fblas", _blas_alias)
