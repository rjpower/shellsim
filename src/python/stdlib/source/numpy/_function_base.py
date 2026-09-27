"""Whole-array helpers that do not reduce to a single statistic: ``gradient``, ``diff``,
``angle``, linear ``interp``/``correlate``/``convolve`` (thin wrappers over the native
``_numpy_math`` kernels), and the condition-driven selectors ``select``, ``extract``, ``place``
and ``putmask``.
"""

import numpy as np
from _numpy_math import _compiled_interp, _compiled_interp_complex, _correlate

__all__ = [
    "angle",
    "convolve",
    "correlate",
    "cross",
    "diff",
    "extract",
    "gradient",
    "interp",
    "place",
    "polyfit",
    "polyval",
    "putmask",
    "select",
    "vecdot",
]


def angle(z, deg=False):
    """The phase of `z` (real input is treated as having a zero imaginary part)."""
    z = np.asanyarray(z)
    if z.dtype.kind == "c":
        result = np.arctan2(z.imag, z.real)
    else:
        result = np.arctan2(np.zeros(z.shape, dtype=np.float64), z.astype(np.float64))
    return result * (180.0 / np.pi) if deg else result


def _diff_edge(a, axis, value):
    """`value` broadcast to a length-1 slab of `a` along `axis`, for `diff`'s prepend/append."""
    value = np.asanyarray(value)
    if value.ndim != 0:
        return value
    shape = list(a.shape)
    shape[axis] = 1
    return np.broadcast_to(value, shape)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """The `n`-th discrete difference of `a` along `axis` (a boolean array uses ``!=``)."""
    a = np.asanyarray(a)
    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")
    axis = axis % a.ndim if a.ndim else axis
    if prepend is not None:
        a = np.concatenate([_diff_edge(a, axis, prepend), a], axis=axis)
    if append is not None:
        a = np.concatenate([a, _diff_edge(a, axis, append)], axis=axis)
    for _ in range(n):
        upper = [slice(None)] * a.ndim
        lower = [slice(None)] * a.ndim
        upper[axis] = slice(1, None)
        lower[axis] = slice(None, -1)
        upper_part, lower_part = a[tuple(upper)], a[tuple(lower)]
        a = (upper_part != lower_part) if a.dtype == np.bool_ else (upper_part - lower_part)
    return a


def gradient(f, *varargs, axis=None, edge_order=1):
    """The central-difference gradient of `f` along `axis` (every axis, by default).

    One array per axis, or a single array when only one axis is differentiated. `varargs` gives
    a uniform spacing per differentiated axis (one value, or one per axis); non-uniform
    coordinate arrays are not supported.
    """
    if edge_order != 1:
        raise NotImplementedError("np.gradient only supports edge_order=1")
    f = np.asanyarray(f)
    if axis is None:
        axes = tuple(range(f.ndim))
    elif isinstance(axis, (int, np.integer)):
        axes = (int(axis) % f.ndim,)
    else:
        axes = tuple(int(ax) % f.ndim for ax in axis)

    if len(varargs) == 0:
        spacings = [1.0] * len(axes)
    elif len(varargs) == 1:
        spacings = [varargs[0]] * len(axes)
    elif len(varargs) == len(axes):
        spacings = list(varargs)
    else:
        raise TypeError("invalid number of arguments")

    dtype = f.dtype if f.dtype.kind in "fc" else np.float64
    results = []
    for ax, spacing in zip(axes, spacings):
        if not isinstance(spacing, (int, float, np.number)):
            raise NotImplementedError("np.gradient only supports scalar, uniform spacing")
        n = f.shape[ax]
        result = np.empty(f.shape, dtype=dtype)
        if n == 1:
            result[...] = 0.0
            results.append(result)
            continue
        center, lower, upper = [slice(None)] * f.ndim, [slice(None)] * f.ndim, [slice(None)] * f.ndim
        center[ax], lower[ax], upper[ax] = slice(1, -1), slice(0, -2), slice(2, None)
        result[tuple(center)] = (f[tuple(upper)].astype(dtype) - f[tuple(lower)].astype(dtype)) / 2.0
        first, second = [slice(None)] * f.ndim, [slice(None)] * f.ndim
        first[ax], second[ax] = 0, 1
        result[tuple(first)] = f[tuple(second)].astype(dtype) - f[tuple(first)].astype(dtype)
        last, before_last = [slice(None)] * f.ndim, [slice(None)] * f.ndim
        last[ax], before_last[ax] = -1, -2
        result[tuple(last)] = f[tuple(last)].astype(dtype) - f[tuple(before_last)].astype(dtype)
        results.append(result / spacing)
    return results[0] if len(results) == 1 else results


def interp(x, xp, fp, left=None, right=None, period=None):
    """Piecewise-linear interpolation of `x` against the samples (`xp`, `fp`).

    With `period`, `x` and `xp` are reduced modulo the period, `xp` is sorted, and one sample
    from each end is repeated one period away so points near the wrap interpolate across it.
    `left` and `right` are ignored.
    """
    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = abs(period)
        x = np.mod(x, period)
        xp = np.mod(xp, period)
        order = np.argsort(xp)
        xp = xp[order]
        fp = np.asanyarray(fp)[order]
        xp = np.concatenate((xp[-1:] - period, xp, xp[0:1] + period))
        fp = np.concatenate((fp[-1:], fp, fp[0:1]))
        left = None
        right = None
    if np.asanyarray(fp).dtype.kind == "c":
        return _compiled_interp_complex(x, xp, fp, left, right)
    return _compiled_interp(x, xp, fp, left, right)


def correlate(a, v, mode="valid"):
    """The cross-correlation of `a` and `v`; complex `v` is conjugated."""
    return _correlate(a, v, mode, True)


def convolve(a, v, mode="full"):
    """The discrete convolution of `a` and `v`: correlation of `a` with `v` reversed."""
    v = np.asanyarray(v)
    return _correlate(a, v[::-1], mode, False)


def polyfit(x, y, deg):
    """The degree-`deg` polynomial's coefficients (highest power first) that least-squares fit
    the points (`x`, `y`), via :func:`numpy.linalg.lstsq` on the Vandermonde matrix of `x`.
    """
    x = np.asanyarray(x, dtype=np.float64)
    y = np.asanyarray(y, dtype=np.float64)
    vander = np.stack([x**power for power in range(deg, -1, -1)], axis=-1)
    coefficients, _, _, _ = np.linalg.lstsq(vander, y, rcond=None)
    return coefficients


def polyval(p, x):
    """The polynomial with coefficients `p` (highest power first) evaluated at `x`."""
    p = np.asanyarray(p)
    result = 0
    for coefficient in p:
        result = result * x + coefficient
    return result


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """The 3-vector cross product of `a` and `b` along their last axis (or `axis`)."""
    if axis is not None:
        axisa = axisb = axisc = axis
    a = np.moveaxis(np.asanyarray(a), axisa, -1)
    b = np.moveaxis(np.asanyarray(b), axisb, -1)
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("incompatible dimensions for cross product (dimension must be 3)")
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    result = np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)
    return np.moveaxis(result, -1, axisc)


def vecdot(a, b, axis=-1):
    """The dot product of `a` and `b` along `axis`, conjugating `a` (as for a complex inner product)."""
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    if a.shape[axis] != b.shape[axis]:
        raise ValueError(
            "vecdot: core dimension mismatch, with gufunc signature (n),(n)->() "
            f"(size {b.shape[axis]} is different from {a.shape[axis]})"
        )
    left = np.conjugate(a) if a.dtype.kind == "c" else a
    return np.sum(left * b, axis=axis)


def select(condlist, choicelist, default=0):
    """`choicelist[i]` at positions where `condlist[i]` is the first true condition, else `default`."""
    if len(condlist) != len(choicelist):
        raise ValueError("list of cases must be same length as list of conditions")
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")
    for index, cond in enumerate(condlist):
        if np.asanyarray(cond).dtype != np.bool_:
            raise TypeError(f"invalid entry {index} in condlist: should be boolean ndarray")
    result = np.asanyarray(default)
    for cond, choice in zip(reversed(condlist), reversed(choicelist)):
        result = np.where(np.asanyarray(cond), choice, result)
    return result


def extract(condition, arr):
    """The elements of (raveled) `arr` at the positions where (raveled) `condition` is nonzero."""
    condition = np.asanyarray(condition)
    arr = np.asanyarray(arr)
    return arr.reshape(-1)[np.flatnonzero(condition.reshape(-1))]


def _require_safe_cast(source_dtype, dest_dtype):
    """Raise as NumPy does when `place`/`putmask` are given a real array of values: unlike a
    plain boolean-mask assignment (which casts permissively), both functions insist the value
    array's dtype cast to the destination's dtype under the 'safe' rule.
    """
    if not np.can_cast(source_dtype, dest_dtype, casting="safe"):
        raise TypeError(
            f"Cannot cast array data from {source_dtype!r} to {dest_dtype!r} according to the rule 'safe'"
        )


def place(arr, mask, vals):
    """Write `vals`, cycled, into `arr` at the positions where `mask` is true, in C order."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"argument 1 must be numpy.ndarray, not {type(arr).__name__}")
    mask = np.asanyarray(mask, dtype=np.bool_)
    if mask.size != arr.size:
        raise ValueError("mask and data must be the same size")
    mask = mask.reshape(arr.shape)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return None
    if isinstance(vals, np.ndarray):
        pool = vals.reshape(-1)
        if pool.size == 0:
            raise ValueError("Cannot insert from an empty array!")
        _require_safe_cast(pool.dtype, arr.dtype)
        cycled = np.take(pool, np.arange(count) % pool.size)
    else:
        pool = list(vals)
        if len(pool) == 0:
            raise ValueError("Cannot insert from an empty array!")
        cycled = [pool[i % len(pool)] for i in range(count)]
    arr[mask] = cycled
    return None


def putmask(a, mask, values):
    """Write `values`, cycled by flat position (not by the count of true positions), where `mask` is true."""
    if not isinstance(a, np.ndarray):
        raise TypeError("putmask: first argument must be an array")
    mask = np.asanyarray(mask, dtype=np.bool_)
    if mask.size != a.size:
        raise ValueError("putmask: mask and data must be the same size")
    if not a.flags.writeable:
        raise ValueError("putmask: output array is read-only")
    mask = mask.reshape(a.shape)
    positions = np.flatnonzero(mask.reshape(-1))
    if isinstance(values, np.ndarray):
        pool = values.reshape(-1)
        if pool.size == 0:
            return None
        _require_safe_cast(pool.dtype, a.dtype)
        selected = np.take(pool, positions % pool.size)
    else:
        pool = values if isinstance(values, (list, tuple)) else [values]
        if len(pool) == 0:
            return None
        selected = [pool[int(index) % len(pool)] for index in positions]
    a[mask] = selected
    return None
