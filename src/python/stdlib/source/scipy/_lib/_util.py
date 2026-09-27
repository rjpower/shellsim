"""Array helpers SciPy's Python code shares, following SciPy 1.18.

SciPy writes these helpers against the array API so that they accept NumPy, CuPy, JAX and other
namespaces. shellsim only has NumPy, so each helper is the NumPy branch of SciPy's code:
``_lazyselect`` and ``check_random_state`` from ``scipy/_lib/_util.py``, ``apply_where`` from
``array_api_extra``, and ``_result_type``, ``_promote`` and ``_count_nonmasked`` from
``xp_result_type``, ``xp_promote`` and ``_count_nonmasked`` in ``scipy/_lib/_array_api.py``.
Masked arrays are not supported, so nothing here looks for masks.

``_asarray_validated``, ``_deprecate_dtypes`` and ``_apply_over_batch`` are SciPy's helpers for
``scipy.linalg``. shellsim has no sparse or masked arrays, so ``_asarray_validated`` checks for
neither. Functions have no writable attributes in shellsim, so ``_apply_over_batch`` returns an
undecorated wrapper where SciPy copies the wrapped function's name and docstring onto it.

``getfullargspec_no_self`` reads parameters through shellsim's private ``_shellsim_introspect``
helper where SciPy calls ``inspect.signature``. shellsim records no annotations, so its
``annotations`` are always empty.
"""

import math
import warnings

import numpy as np
from _shellsim_introspect import parameters as _parameters
from scipy._lib._bunch import _make_tuple_bunch

__all__ = [
    "_apply_over_batch",
    "_asarray_validated",
    "_contains_nan",
    "_deprecate_dtypes",
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


def _asarray_validated(
    a, check_finite=True, sparse_ok=False, objects_ok=False, mask_ok=False, as_inexact=False
):
    """``a`` as an array, rejecting non-finite values when ``check_finite`` and object arrays
    unless ``objects_ok``. With ``as_inexact``, non-floating input becomes ``float64``.
    """
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok and a.dtype == np.dtype("O"):
        raise ValueError("object arrays are not supported")
    if as_inexact and not np.issubdtype(a.dtype, np.inexact):
        a = toarray(a, dtype=np.float64)
    return a


def _deprecate_dtypes(func_name, *arrays):
    """Warn, once per call, about an array whose dtype SciPy 1.18 deprecates for linalg.

    Integers and ``float32``, ``float64``, ``complex64`` and ``complex128`` are accepted
    silently; ``None`` entries are skipped.
    """
    for a in arrays:
        if a is None:
            continue
        if a.dtype.char not in np.typecodes["AllInteger"] + "fdFD":
            msg = (
                f"Calling {func_name} with arguments of dtype={a.dtype} "
                f"({a.dtype.char = }) is deprecated in SciPy 1.18.0 and "
                "will be removed in SciPy 1.20.0. Please cast array inputs to "
                "one of np.float{32,64} or np.complex{64,128} manually."
            )
            warnings.warn(msg, category=DeprecationWarning, stacklevel=3)
            return


def _apply_over_batch(*argdefs):
    """Decorate a function of core-shaped arrays so that it loops over leading batch axes.

    Each ``argdef`` is ``(name, ndim)``: the leading positional or keyword argument ``name`` has
    ``ndim`` core axes, or ``"1|2"`` for a right-hand side that may be a vector or a matrix.
    Unbatched calls go straight through. Batched calls broadcast the batch shapes, call the
    function once per batch element and stack each output; several outputs come back as a list.

    >>> @_apply_over_batch(("a", 2))
    ... def trace(a):
    ...     return np.trace(a)
    >>> trace(np.ones((3, 2, 2))).shape
    (3,)
    """
    names, ndims = list(zip(*argdefs))
    n_arrays = len(names)

    def decorator(f):
        def wrapper(*args, **kwargs):
            args = list(args)
            arrays, other_args = args[:n_arrays], args[n_arrays:]
            for i, name in enumerate(names):
                if name in kwargs:
                    if i + 1 <= len(args):
                        raise ValueError(
                            f"{f.__name__}() got multiple values for argument `{name}`."
                        )
                    arrays.append(kwargs.pop(name))

            batch_shapes = []
            core_shapes = []
            for i, (array, ndim) in enumerate(zip(arrays, ndims)):
                array = None if array is None else np.asarray(array)
                shape = () if array is None else array.shape
                if ndim == "1|2":
                    ndim = 2 if array.ndim >= 2 else 1
                arrays[i] = array
                batch_shapes.append(shape[:-ndim] if ndim > 0 else shape)
                core_shapes.append(shape[-ndim:] if ndim > 0 else ())

            _deprecate_dtypes(f.__name__, *arrays)

            if not any(batch_shapes):
                return f(*arrays, *other_args, **kwargs)

            batch_shape = np.broadcast_shapes(*batch_shapes)
            if math.prod(batch_shape) == 0:
                raise ValueError(f"`{f.__name__}` does not support zero-size batches.")
            for i, (array, core_shape) in enumerate(zip(arrays, core_shapes)):
                if array is not None:
                    arrays[i] = np.broadcast_to(array, batch_shape + core_shape)

            results = []
            for index in np.ndindex(batch_shape):
                result = f(
                    *((array[index] if array is not None else None) for array in arrays),
                    *other_args,
                    **kwargs,
                )
                results.append(result if isinstance(result, tuple) else (result,))
            results = list(zip(*results))
            for i, result in enumerate(results):
                result = np.stack(result)
                results[i] = np.reshape(result, batch_shape + result.shape[1:])
            return results[0] if len(results) == 1 else results

        return wrapper

    return decorator
