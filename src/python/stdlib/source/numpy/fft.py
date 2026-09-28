"""Discrete Fourier transforms: ``fft``, ``ifft``, ``rfft``, ``irfft``, their two- and
n-dimensional forms (``fft2``, ``ifft2``, ``fftn``, ``ifftn``), ``fftfreq``, ``rfftfreq``,
``fftshift``, and ``ifftshift``. ``hfft``, ``ihfft``, and the real n-dimensional transforms
(``rfft2``, ``irfft2``, ``rfftn``, ``irfftn``) are not implemented.

Every transform reduces to one O(n log n) kernel, ``_dft1d``, applied along one axis of a
(batch, n) view built with `np.moveaxis` and `np.reshape`: a power-of-two `n` runs an iterative
radix-2 Cooley-Tukey transform (bit-reversal permutation, then log2(n) vectorized butterfly
stages); any other `n` runs Bluestein's chirp-z transform, which rewrites the length-n DFT as a
convolution computed with two more radix-2 transforms padded to the next power of two. Every
stage is a handful of whole-array NumPy calls batched over every row at once, so the only Python
loops here run once per stage (`log2(n)` of them) or once per axis, never once per element.

``float16``/``float32``/``complex64`` input computes at double precision and rounds back to
single precision (``complex64``/``float32``) only in the returned array, matching NumPy's own
single-precision dispatch without a second, single-precision code path.
"""

import numpy as np

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]

# float16 and float32 real input, and complex64 input, keep single precision; everything else
# (integers, bool, float64, complex128) computes and returns double precision.
_SINGLE_PRECISION = (np.float16, np.float32, np.complex64)


def _complex_dtype_for(a):
    return np.complex64 if a.dtype in _SINGLE_PRECISION else np.complex128


def _real_dtype_for(a):
    return np.float32 if a.dtype in _SINGLE_PRECISION else np.float64


def _require_numeric(a, name):
    if a.dtype.kind not in "biufc":
        raise TypeError(f"{name} needs numeric input")


def _reject_complex(a, name):
    if a.dtype.kind == "c":
        raise TypeError(f"{name} needs real input")


def _resolve_axis(shape, axis):
    """Normalize `axis` (default -1) against `shape` by indexing the shape tuple with it, so a
    non-integer `axis` raises `TypeError` and an out-of-range one raises `IndexError`, exactly as
    plain tuple indexing does (NumPy's `fft` resolves `axis` the same way, rather than through its
    usual `AxisError`).
    """
    if axis is None:
        axis = -1
    _ = shape[axis]
    return int(axis) % len(shape)


def _resolve_n(n, default):
    """`n=` resolved against `default`; the result must be positive."""
    if n is None:
        n = default
    elif not isinstance(n, (int, np.integer)):
        raise TypeError(f"'{type(n).__name__}' object cannot be interpreted as an integer")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return n


def _norm_scale(n, norm, inverse):
    """The multiplicative factor `norm=` applies once, after the unnormalized transform."""
    if norm is None or norm == "backward":
        return 1.0 / n if inverse else 1.0
    if norm == "ortho":
        return 1.0 / np.sqrt(n)
    if norm == "forward":
        return 1.0 if inverse else 1.0 / n
    raise ValueError(f"norm must be 'backward', 'ortho' or 'forward', not {norm!r}")


def _scale_complex(x, factor):
    """`x * factor` for a real `factor`, computed componentwise (`x.real*factor`,
    `x.imag*factor`) rather than through the general complex-times-complex formula. That formula
    promotes `factor` to `factor+0j` and computes the imaginary part as `x.real*0 + x.imag*factor`,
    so an infinite `x.real` turns the unused `x.real*0` cross term into a spurious `nan` (and a
    `RuntimeWarning`) even though a real scale should never touch the imaginary part that way.
    """
    if factor == 1.0:
        return x
    return x.real * factor + 1j * (x.imag * factor)


def _finish(name, result, out):
    """Return `result`, or copy it into `out` (which must have the same shape and a same-kind
    castable dtype) and return `out`.
    """
    if out is None:
        return result
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be an array")
    if out.shape != result.shape:
        raise ValueError("out has the wrong shape")
    if not np.can_cast(result.dtype, out.dtype, casting="same_kind"):
        raise TypeError(f"cannot store {name}'s {result.dtype} result in {out.dtype} out")
    out[...] = result
    return out


def _resize_last_axis(x, n):
    """`x` cropped or zero-padded along its last axis to length `n`."""
    old_n = x.shape[-1]
    if n == old_n:
        return x
    if n < old_n:
        return x[..., :n]
    pad = np.zeros(x.shape[:-1] + (n - old_n,), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def _bit_reversal_permutation(n):
    """The length-`n` permutation `p` with `p[i]` equal to `i`'s bits reversed (`n` a power of
    two), built by `log2(n)` vectorized doublings rather than a loop over its `n` entries:
    `rev(2k)` is `rev(k)` written out twice, once with a leading 0 bit and once with a leading 1
    bit, i.e. `concatenate([2*rev(k), 2*rev(k)+1])`.
    """
    perm = np.zeros(1, dtype=np.int64)
    size = 1
    while size < n:
        perm = np.concatenate([2 * perm, 2 * perm + 1])
        size *= 2
    return perm


def _radix2(x, inverse):
    """In-place-style iterative radix-2 Cooley-Tukey transform of the last axis of `x` (any
    leading shape), unnormalized in both directions. `x.shape[-1]` must be a power of two.
    """
    n = x.shape[-1]
    if n <= 1:
        return x.copy()
    x = np.take(x, _bit_reversal_permutation(n), axis=-1)
    turn = 1j if inverse else -1j
    size = 2
    while size <= n:
        half = size // 2
        twiddle = np.exp(turn * 2 * np.pi * np.arange(half) / size)
        blocks = x.reshape(x.shape[:-1] + (n // size, size))
        a = blocks[..., :half]
        b = blocks[..., half:] * twiddle
        x = np.concatenate([a + b, a - b], axis=-1).reshape(x.shape[:-1] + (n,))
        size *= 2
    return x


def _bluestein(x, inverse):
    """Bluestein's chirp-z transform for a length `n = x.shape[-1]` that is not a power of two:
    `X[k] = chirp[k] * conv(x*chirp, conj(chirp))[k]`, where `chirp[k] = exp(sign*i*pi*k^2/n)`
    and the length-`n` linear convolution runs as a circular convolution at the next power of
    two `m >= 2n-1`, via two more `_radix2` transforms.
    """
    n = x.shape[-1]
    # `2*n - 1` is always odd, so it is a power of two only when it equals 1 (n == 1, handled
    # before this function ever runs); `bit_length` therefore always finds the next one above it.
    m = 1 << (2 * n - 1).bit_length()
    sign = 1.0 if inverse else -1.0
    k = np.arange(n, dtype=np.float64)
    chirp = np.exp(sign * 1j * np.pi * (k * k) / n)
    a = np.zeros(x.shape[:-1] + (m,), dtype=np.complex128)
    a[..., :n] = x * chirp
    conj_chirp = np.conj(chirp)
    b = np.zeros(m, dtype=np.complex128)
    b[:n] = conj_chirp
    mirror = n - 1
    if mirror > 0:
        b[m - mirror :] = conj_chirp[1:][::-1]
    convolved = _radix2(_radix2(a, False) * _radix2(b, False), True) / m
    return convolved[..., :n] * chirp


def _dft1d(x, inverse):
    """The unnormalized O(n log n) DFT of the last axis of `x`: forward (`inverse=False`)
    computes `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)`; `inverse=True` computes the same sum with
    a `+` sign, and neither direction divides by `n`.
    """
    n = x.shape[-1]
    if n <= 1:
        return x.copy()
    if n & (n - 1) == 0:
        return _radix2(x, inverse)
    return _bluestein(x, inverse)


def _dft_axis(x, n, axis, inverse, out_len=None):
    """Run [`_dft1d`] along `axis` of the complex array `x`: crop or zero-pad to length `n`
    first, then (only for `rfft`'s one-sided output) keep just the first `out_len` of the `n`
    transformed bins.
    """
    x = np.moveaxis(x, axis, -1)
    x = _resize_last_axis(x, n)
    shape = x.shape
    batch = 1
    for dim in shape[:-1]:
        batch *= dim
    flat = _dft1d(x.reshape(batch, n), inverse)
    if out_len is not None and out_len != n:
        flat = flat[:, :out_len]
    result = flat.reshape(shape[:-1] + (flat.shape[-1],))
    return np.moveaxis(result, -1, axis)


def _forward_or_inverse(a, n, axis, norm, out, inverse, name):
    a = np.asarray(a)
    _require_numeric(a, name)
    axis = _resolve_axis(a.shape, axis)
    n = _resolve_n(n, a.shape[axis])
    scale = _norm_scale(n, norm, inverse)
    out_dtype = _complex_dtype_for(a)
    x = _dft_axis(a.astype(np.complex128), n, axis, inverse)
    result = _scale_complex(x, scale).astype(out_dtype)
    return _finish(name, result, out)


def fft(a, n=None, axis=-1, norm=None, out=None):
    """The discrete Fourier transform of `a` along `axis` (last by default)."""
    return _forward_or_inverse(a, n, axis, norm, out, False, "fft")


def ifft(a, n=None, axis=-1, norm=None, out=None):
    """The inverse of :func:`fft`."""
    return _forward_or_inverse(a, n, axis, norm, out, True, "ifft")


def rfft(a, n=None, axis=-1, norm=None, out=None):
    """The discrete Fourier transform of the real array `a`, keeping only the `n//2+1`
    non-redundant bins of its Hermitian-symmetric spectrum.
    """
    a = np.asarray(a)
    _reject_complex(a, "rfft")
    _require_numeric(a, "rfft")
    axis = _resolve_axis(a.shape, axis)
    n = _resolve_n(n, a.shape[axis])
    scale = _norm_scale(n, norm, False)
    out_dtype = _complex_dtype_for(a)
    x = _dft_axis(a.astype(np.complex128), n, axis, False, out_len=n // 2 + 1)
    result = _scale_complex(x, scale).astype(out_dtype)
    return _finish("rfft", result, out)


def _rebuild_hermitian(x, axis, n):
    """The one-sided spectrum `x` (`n//2+1` bins along `axis`) extended to the full length-`n`
    Hermitian-symmetric spectrum `irfft`'s transform expects: the given bins keep their place and
    `full[n-k] = conj(x[k])` mirrors the rest.
    """
    m = n // 2 + 1
    x = np.moveaxis(x, axis, -1)
    x = _resize_last_axis(x, m)
    mirror = n - m
    if mirror > 0:
        tail = np.conj(x[..., 1 : 1 + mirror][..., ::-1])
        x = np.concatenate([x, tail], axis=-1)
    return np.moveaxis(x, -1, axis)


def irfft(a, n=None, axis=-1, norm=None, out=None):
    """The inverse of :func:`rfft`. `n` (default `2*(a.shape[axis]-1)`) is the length of the
    real output, not of `a`.
    """
    a = np.asarray(a)
    _require_numeric(a, "irfft")
    axis = _resolve_axis(a.shape, axis)
    n = _resolve_n(n, 2 * (a.shape[axis] - 1))
    scale = _norm_scale(n, norm, True)
    out_dtype = _real_dtype_for(a)
    x = _rebuild_hermitian(a.astype(np.complex128), axis, n)
    result = (_dft_axis(x, n, axis, True).real * scale).astype(out_dtype)
    return _finish("irfft", result, out)


def _int_sequence(value):
    """`s=`/`axes=` must be a tuple or list of ints; unlike `n=`/`axis=`, a bare int is rejected
    the same way Python rejects iterating over one.
    """
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"'{type(value).__name__}' object is not iterable")
    return [_index_of(item) for item in value]


def _index_of(value):
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"'{type(value).__name__}' object cannot be interpreted as an integer")
    return int(value)


def _positive_length(value):
    if value <= 0:
        raise ValueError(f"n must be positive, got {value}")
    return value


def _resolve_nd(shape, s, axes):
    """Resolve `s=`/`axes=` to normalized axes and their target lengths. With `axes=None`, the
    axes are `s`'s trailing `len(s)` axes when only `s` is given, or every axis otherwise
    (`fft2`/`ifft2` never reach the `axes=None` case except when a caller passes it explicitly,
    since their own `axes` parameter otherwise defaults to `(-2, -1)`).
    """
    raw_s = _int_sequence(s) if s is not None else None
    if axes is not None:
        axes_list = _int_sequence(axes)
    elif raw_s is not None:
        axes_list = list(range(-len(raw_s), 0))
    else:
        axes_list = list(range(len(shape)))
    identity = tuple(range(len(shape)))
    normalized_axes = [identity[axis] for axis in axes_list]
    if raw_s is None:
        lengths = [shape[axis] for axis in normalized_axes]
    else:
        if len(raw_s) != len(normalized_axes):
            raise ValueError("s and axes must have the same length")
        lengths = [
            shape[axis] if value == -1 else _positive_length(value)
            for value, axis in zip(raw_s, normalized_axes)
        ]
    return normalized_axes, lengths


def _fftn_impl(a, s, axes, norm, out, inverse, name):
    a = np.asarray(a)
    _require_numeric(a, name)
    normalized_axes, lengths = _resolve_nd(a.shape, s, axes)
    out_dtype = _complex_dtype_for(a)
    x = a.astype(np.complex128)
    for axis, n in zip(normalized_axes, lengths):
        x = _scale_complex(_dft_axis(x, n, axis, inverse), _norm_scale(n, norm, inverse))
    result = x.astype(out_dtype)
    return _finish(name, result, out)


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """`fft` applied over each of `axes` (last two by default) in turn."""
    return _fftn_impl(a, s, axes, norm, out, False, "fft")


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """The inverse of :func:`fft2`."""
    return _fftn_impl(a, s, axes, norm, out, True, "ifft")


def fftn(a, s=None, axes=None, norm=None, out=None):
    """`fft` applied over each of `axes` (every axis by default) in turn."""
    return _fftn_impl(a, s, axes, norm, out, False, "fft")


def ifftn(a, s=None, axes=None, norm=None, out=None):
    """The inverse of :func:`fftn`."""
    return _fftn_impl(a, s, axes, norm, out, True, "ifft")


def _resolve_freq_n(n):
    """`n` must be a non-negative integer; `n=0` raises `ZeroDivisionError` where it is divided
    by.
    """
    if not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    return n


def fftfreq(n, d=1.0):
    """The `n` sample frequencies for a length-`n` FFT with sample spacing `d`, ordered
    `[0, 1, ..., n//2-1, -(n//2), ..., -1] / (n*d)` (positive frequencies first, as `fft`'s own
    output bins are).
    """
    n = _resolve_freq_n(n)
    scale = 1.0 / (n * d)
    k = np.arange(n)
    half = (n + 1) // 2
    return np.where(k < half, k, k - n).astype(np.float64) * scale


def rfftfreq(n, d=1.0):
    """The `n//2+1` non-negative sample frequencies :func:`rfft` keeps, `[0, 1, ..., n//2] /
    (n*d)`.
    """
    n = _resolve_freq_n(n)
    scale = 1.0 / (n * d)
    return np.arange(n // 2 + 1, dtype=np.float64) * scale


def _shift(x, axes, sign):
    x = np.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, (int, np.integer)):
        axes = (int(axes),)
    else:
        axes = tuple(int(axis) for axis in axes)
    shifts = tuple(sign * (x.shape[axis] // 2) for axis in axes)
    return np.roll(x, shifts, axis=axes)


def fftshift(x, axes=None):
    """`x` with its zero-frequency bin (`axes` default: every axis) moved to the middle, by
    rolling each axis forward by half its length.
    """
    return _shift(x, axes, 1)


def ifftshift(x, axes=None):
    """The inverse of :func:`fftshift`."""
    return _shift(x, axes, -1)
