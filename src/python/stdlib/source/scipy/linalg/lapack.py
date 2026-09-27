"""shellsim's ``scipy.linalg.lapack``, following SciPy 1.18's ``scipy/linalg/lapack.py``.

SciPy exposes its f2py-compiled LAPACK wrappers here. shellsim provides the real (``s`` and
``d``) forms of ``getrf``, ``getrs``, ``gecon``, ``getri``, ``trtrs``, ``trtri``, ``potrf``,
``potrs``, ``potri``, ``gtsv``, ``gbsv`` and ``lange``, computed by the native
``_scipy_linalg`` module from ports of reference LAPACK 3.12. Other routines, and every complex
routine, raise ``NotImplementedError``.

The routines take f2py's arguments and defaults. ``overwrite_*`` flags are accepted and ignored,
since results are always new arrays, and workspace sizes such as ``lwork`` are ignored because
the ports allocate their own. Pivots are 0-based ``int32`` arrays, as f2py returns them.

The module also holds the dtype helpers that SciPy's ``scipy.linalg`` functions share.
"""

import numpy as np
import _scipy_linalg as _native
from scipy.linalg.blas import (
    HAS_ILP64,
    HAS_LP64,
    _as,
    _define,
    _get_funcs,
    _resolve_ilp64,
    _unsupported,
)
from scipy.linalg.blas import find_best_blas_type as find_best_lapack_type

__all__ = ["get_lapack_funcs"]


class error(Exception):
    """The exception f2py raises when an argument fails its check, ``_flapack.error``."""

    __module__ = "_flapack"


def _check_trans(dtype, name, position, trans):
    """f2py's range check on the ``trans`` keyword of ``getrs`` and ``trtrs``."""
    trans = int(trans)
    if not 0 <= trans <= 2:
        routine = f"{'d' if dtype == np.float64 else 's'}{name}"
        raise error(
            f"(trans>=0 && trans <=2) failed for {position} keyword trans: {routine}:trans={trans}"
        )
    return trans


def _getrf(dtype, a, overwrite_a=0):
    return _native.getrf(_as(a, dtype))


def _getrs(dtype, lu, piv, b, trans=0, overwrite_b=0):
    trans = _check_trans(dtype, "getrs", "1st", trans)
    return _native.getrs(_as(lu, dtype), _as(piv, np.int32), _as(b, dtype), trans)


def _gecon(dtype, a, anorm, norm="1"):
    return _native.gecon(_as(a, dtype), float(anorm), norm)


def _getri(dtype, lu, piv, lwork=None, overwrite_lu=0):
    return _native.getri(_as(lu, dtype), _as(piv, np.int32))


def _trtrs(dtype, a, b, lower=0, trans=0, unitdiag=0, lda=None, overwrite_b=0):
    trans = _check_trans(dtype, "trtrs", "2nd", trans)
    return _native.trtrs(_as(a, dtype), _as(b, dtype), bool(lower), trans, bool(unitdiag))


def _trtri(dtype, c, lower=0, unitdiag=0, overwrite_c=0):
    return _native.trtri(_as(c, dtype), bool(lower), bool(unitdiag))


def _potrf(dtype, a, lower=0, clean=1, overwrite_a=0):
    return _native.potrf(_as(a, dtype), bool(lower), bool(clean))


def _potrs(dtype, c, b, lower=0, overwrite_b=0):
    return _native.potrs(_as(c, dtype), _as(b, dtype), bool(lower))


def _potri(dtype, c, lower=0, overwrite_c=0):
    return _native.potri(_as(c, dtype), bool(lower))


def _gtsv(dtype, dl, d, du, b, overwrite_dl=0, overwrite_d=0, overwrite_du=0, overwrite_b=0):
    return _native.gtsv(_as(dl, dtype), _as(d, dtype), _as(du, dtype), _as(b, dtype))


def _gbsv(dtype, kl, ku, ab, b, overwrite_ab=0, overwrite_b=0):
    return _native.gbsv(int(kl), int(ku), _as(ab, dtype), _as(b, dtype))


def _lange(dtype, norm, a):
    return _native.lange(norm, _as(a, dtype))


_ROUTINES = _define(
    {
        "getrf": _getrf,
        "getrs": _getrs,
        "gecon": _gecon,
        "getri": _getri,
        "trtrs": _trtrs,
        "trtri": _trtri,
        "potrf": _potrf,
        "potrs": _potrs,
        "potri": _potri,
        "gtsv": _gtsv,
        "gbsv": _gbsv,
        "lange": _lange,
    },
    "flapack",
)

_lapack_alias = {
    "corghr": "cunghr",
    "zorghr": "zunghr",
    "corghr_lwork": "cunghr_lwork",
    "zorghr_lwork": "zunghr_lwork",
    "corgqr": "cungqr",
    "zorgqr": "zungqr",
    "cormqr": "cunmqr",
    "zormqr": "zunmqr",
    "corgrq": "cungrq",
    "zorgrq": "zungrq",
}


def __getattr__(name):
    # The routines are served from here because shellsim has no `globals()` to define them with.
    if name in _ROUTINES:
        return _ROUTINES[name]
    if len(name) > 1 and name[0] in "sdcz":
        raise _unsupported("LAPACK", name)
    raise AttributeError(f"module 'scipy.linalg.lapack' has no attribute '{name}'")


def get_lapack_funcs(names, arrays=(), dtype=None, ilp64="preferred"):
    """LAPACK routines by name, in the precision that ``arrays`` or ``dtype`` call for.

    A single name gives a single routine; a sequence of names gives a list.

    >>> getrf, = get_lapack_funcs(("getrf",), (np.eye(2),))
    >>> getrf.typecode
    'd'
    """
    _resolve_ilp64("LAPACK", ilp64)
    return _get_funcs(names, arrays, dtype, "LAPACK", _ROUTINES, "flapack", _lapack_alias)


def _normalize_lapack_dtype(a, overwrite_a):
    """``a`` in the LAPACK precision its dtype maps to, and whether it may now be overwritten."""
    _, dtyp, _ = find_best_lapack_type((a,))
    needs_copy = dtyp.char != a.dtype.char
    if needs_copy:
        a = a.astype(dtyp)
    return a, overwrite_a or needs_copy


def _normalize_lapack_dtype1(a, overwrite_a):
    """``a`` cast to the first LAPACK dtype it casts to safely, as ``det`` needs."""
    if a.dtype.char not in "fdFD":
        dtype_char = "".join([y for y in "fdFD" if np.can_cast(a.dtype, y)])
        if not dtype_char:
            raise TypeError(
                f"The dtype {a.dtype} cannot be cast to float(32, 64) or complex(64, 128)."
            )
        a = a.astype(dtype_char[0])
        overwrite_a = True
    return a, overwrite_a


def _ensure_dtype_cdsz(*arrays):
    """``arrays`` cast to their common LAPACK dtype: ``float32``, ``float64``, ``complex64``
    or ``complex128``."""
    dtype = np.result_type(*arrays)
    if not np.issubdtype(dtype, np.inexact):
        return (array.astype(np.float64) for array in arrays)
    complex = np.issubdtype(dtype, np.complexfloating)
    if np.finfo(dtype).bits <= 32:
        dtype = np.complex64 if complex else np.float32
    elif np.finfo(dtype).bits >= 64:
        dtype = np.complex128 if complex else np.float64
    return (array.astype(dtype, copy=False) for array in arrays)


def _ensure_aligned_and_native(a, overwrite_a):
    """``a`` unchanged: shellsim's arrays are always aligned and in native byte order."""
    return a, overwrite_a
