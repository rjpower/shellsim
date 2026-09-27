"""Array helpers SciPy's Python code shares, following SciPy 1.18.

SciPy writes these helpers against the array API so that they accept NumPy, CuPy, JAX and other
namespaces. shellsim only has NumPy, so each helper is the NumPy branch of SciPy's code:
``_lazyselect`` and ``check_random_state`` from ``scipy/_lib/_util.py``, ``apply_where`` from
``array_api_extra``, and ``_result_type``, ``_promote`` and ``_count_nonmasked`` from
``xp_result_type``, ``xp_promote`` and ``_count_nonmasked`` in ``scipy/_lib/_array_api.py``.
Masked arrays are not supported, so nothing here looks for masks.

``getfullargspec_no_self`` reads parameters through shellsim's private ``_shellsim_introspect``
helper where SciPy calls ``inspect.signature``. shellsim records no annotations, so its
``annotations`` are always empty.
"""

import numpy as np
from _shellsim_introspect import parameters as _parameters
from scipy._lib._bunch import _make_tuple_bunch

__all__ = [
    "_contains_nan",
    "_count_nonmasked",
    "_get_nan",
    "_lazyselect",
    "_promote",
    "_result_type",
    "apply_where",
    "check_random_state",
    "getfullargspec_no_self",
]


def check_random_state(seed):
    """Turn ``seed`` into a ``RandomState`` or ``Generator``, as SciPy does.

    ``None`` and ``np.random`` give NumPy's global legacy stream, an integer seeds a new
    ``RandomState``, and a ``RandomState`` or ``Generator`` is returned as given.
    """
    if seed is None or seed is np.random:
        return np.random._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    raise ValueError(f"'{seed}' cannot be used to seed a numpy.random.RandomState instance")


def _lazyselect(condlist, choicelist, arrays, default=0):
    """Like ``np.select``, but each choice is a function applied only where its condition holds."""
    arrays = np.broadcast_arrays(*arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=default, dtype=tcode)
    for func, cond in zip(choicelist, condlist):
        if np.all(cond is False):
            continue
        cond, _ = np.broadcast_arrays(cond, arrays[0])
        temp = tuple(np.extract(cond, arr) for arr in arrays)
        np.place(out, cond, func(*temp))
    return out


def apply_where(cond, args, f1, f2=None, *, fill_value=None):
    """Evaluate ``f1`` where ``cond`` holds and ``f2`` (or ``fill_value``) elsewhere.

    Each function sees only the elements it is responsible for, so it never computes, or warns
    about, values that the other branch replaces. ``args`` is an array or a tuple of arrays,
    broadcast against ``cond``.

    >>> apply_where(x > 0, x, np.log, fill_value=0.0)  # log of the positive elements only
    """
    if (f2 is None) == (fill_value is None):
        raise TypeError("Exactly one of `fill_value` or `f2` must be given.")
    args = list(args) if isinstance(args, tuple) else [args]
    if fill_value is None or isinstance(fill_value, (int, float, complex)):
        cond, *args = np.broadcast_arrays(cond, *args)
    else:
        cond, fill_value, *args = np.broadcast_arrays(cond, fill_value, *args)

    temp1 = f1(*(arr[cond] for arr in args))
    if f2 is None:
        dtype = np.result_type(temp1, fill_value)
        if isinstance(fill_value, (int, float, complex)):
            out = np.full_like(cond, dtype=dtype, fill_value=fill_value)
        else:
            out = np.asarray(fill_value).astype(dtype, copy=True)
    else:
        ncond = ~cond
        temp2 = f2(*(arr[ncond] for arr in args))
        dtype = np.result_type(temp1, temp2)
        out = np.empty_like(cond, dtype=dtype)
        out[ncond] = temp2
    out[cond] = temp1
    return out


def _result_type(*args, force_floating=False):
    """The dtype ``args`` promote to, ignoring ``None``; at least ``float64`` if ``force_floating``.

    Iterables become arrays first, so ``1.0`` promotes as a weak Python scalar but ``[1.0]`` as a
    ``float64`` array.
    """
    args = [np.asanyarray(arg) if np.iterable(arg) else arg for arg in args]
    args_not_none = [arg for arg in args if arg is not None]
    if force_floating:
        args_not_none.append(1.0)
    return np.result_type(*args_not_none)


def _promote(*args, broadcast=False, force_floating=False):
    """Convert ``args`` to arrays of their common dtype, leaving ``None`` in place.

    With ``broadcast``, the arrays are also broadcast to a common shape. A single argument is
    returned alone rather than in a tuple.
    """
    if not args:
        return args
    args = [np.asanyarray(arg) if np.iterable(arg) else arg for arg in args]
    dtype = _result_type(*args, force_floating=force_floating)
    args = [np.asanyarray(arg, dtype=dtype) if arg is not None else arg for arg in args]
    if broadcast:
        args_not_none = [arg for arg in args if arg is not None]
        shapes = {arg.shape for arg in args_not_none}
        try:
            shape = np.broadcast_shapes(*shapes) if len(shapes) != 1 else args_not_none[0].shape
        except ValueError as e:
            raise ValueError("Array shapes are incompatible for broadcasting.") from e
        args = [
            arg if arg is None or arg.shape == shape else np.broadcast_to(arg, shape)
            for arg in args
        ]
    return args[0] if len(args) == 1 else tuple(args)


def _count_nonmasked(x, axis, keepdims=False):
    """The number of elements of ``x`` along ``axis`` (an integer, a tuple, or ``None`` for all)."""
    if axis is None:
        return x.size
    return int(np.prod(np.asarray(x.shape)[np.asarray(axis)]))


def _get_nan(*data, shape=()):
    """A NaN, or an array of NaNs of ``shape``, in the floating dtype ``data`` promotes to."""
    dtype = _result_type(*data, force_floating=True)
    res = np.full(shape, np.nan, dtype=dtype)
    if not shape:
        res = res[()]
    return res


def _contains_nan(a, nan_policy="propagate"):
    """Whether ``a`` holds a NaN, after validating ``nan_policy``.

    Raises ``ValueError`` when ``nan_policy`` is ``'raise'`` and there is a NaN. Integer, boolean
    and string arrays cannot hold NaN.
    """
    policies = {"propagate", "raise", "omit"}
    if nan_policy not in policies:
        raise ValueError(f"nan_policy must be one of {policies}.")
    if a.size == 0:
        return False
    if np.isdtype(a.dtype, "real floating"):
        # Unlike most reductions, max returns NaN only when there is a NaN.
        contains_nan = np.isnan(np.max(a))
    elif np.isdtype(a.dtype, "complex floating"):
        contains_nan = np.isnan(np.max(np.real(a))) | np.isnan(np.max(np.imag(a)))
    elif np.issubdtype(a.dtype, object):
        contains_nan = False
        for el in a.ravel():
            if np.issubdtype(type(el), np.number) and np.isnan(el):
                contains_nan = True
                break
    else:
        return False
    if nan_policy == "raise" and contains_nan:
        raise ValueError("The input contains nan values")
    return contains_nan


FullArgSpec = _make_tuple_bunch(
    "FullArgSpec",
    ["args", "varargs", "varkw", "defaults", "kwonlyargs", "kwonlydefaults", "annotations"],
)


def getfullargspec_no_self(func):
    """``inspect.getfullargspec`` without a bound method's ``self``, as SciPy computes it."""
    params = _parameters(func)
    if params is None:
        raise ValueError(f"no signature found for {func!r}")
    args = [name for name, kind, _, _ in params if kind in ("POSITIONAL_OR_KEYWORD", "POSITIONAL_ONLY")]
    varargs = [name for name, kind, _, _ in params if kind == "VAR_POSITIONAL"]
    varargs = varargs[0] if varargs else None
    varkw = [name for name, kind, _, _ in params if kind == "VAR_KEYWORD"]
    varkw = varkw[0] if varkw else None
    defaults = (
        tuple(
            default
            for _, kind, has_default, default in params
            if kind == "POSITIONAL_OR_KEYWORD" and has_default
        )
        or None
    )
    kwonlyargs = [name for name, kind, _, _ in params if kind == "KEYWORD_ONLY"]
    kwdefaults = {
        name: default
        for name, kind, has_default, default in params
        if kind == "KEYWORD_ONLY" and has_default
    }
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwdefaults or None, {})
