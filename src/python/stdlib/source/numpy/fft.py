"""``numpy.fft``, ported from NumPy 2.5's ``numpy/fft/_pocketfft.py`` and ``_helper.py``.

The transforms call the native module ``_numpy_fft``, which stands in for NumPy's
``_pocketfft_umath`` gufuncs and runs a port of the same pocketfft code. Each gufunc takes the
transform length and axis as arguments instead of reading them from the output's shape and the
``axes`` keyword, and always writes into ``out``.
"""

import warnings

import _numpy_fft as pfu
from _numpy_shape import _normalize_axis_index as normalize_axis_index
from numpy import (
    arange,
    asarray,
    conjugate,
    empty,
    empty_like,
    integer,
    reciprocal,
    result_type,
    roll,
    sqrt,
    take,
)

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "rfftn",
    "irfftn",
    "rfft2",
    "irfft2",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
]


def _raw_fft(a, n, axis, is_real, is_forward, norm, out=None):
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    # Calculate the normalization factor, passing in the array dtype to avoid precision loss
    # in the possible sqrt or reciprocal.
    if not is_forward:
        norm = _swap_direction(norm)

    real_dtype = result_type(a.real.dtype, 1.0)
    if norm is None or norm == "backward":
        fct = 1
    elif norm == "ortho":
        fct = reciprocal(sqrt(n, dtype=real_dtype))
    elif norm == "forward":
        fct = reciprocal(n, dtype=real_dtype)
    else:
        raise ValueError(
            f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
        )

    n_out = n
    if is_real:
        if is_forward:
            ufunc = pfu.rfft_n_even if n % 2 == 0 else pfu.rfft_n_odd
            n_out = n // 2 + 1
        else:
            ufunc = pfu.irfft
    else:
        ufunc = pfu.fft if is_forward else pfu.ifft

    axis = normalize_axis_index(axis, a.ndim)

    if out is None:
        if is_real and not is_forward:  # irfft, complex in, real output.
            out_dtype = real_dtype
        else:  # Others, complex output.
            out_dtype = result_type(a.dtype, 1j)
        out = empty_like(a, shape=a.shape[:axis] + (n_out,) + a.shape[axis + 1 :], dtype=out_dtype)
    elif (shape := getattr(out, "shape", None)) is not None and (
        len(shape) != a.ndim or shape[axis] != n_out
    ):
        raise ValueError("output array has wrong shape.")

    return ufunc(a, fct, axis, n, out)


_SWAP_DIRECTION_MAP = {
    "backward": "forward",
    None: "forward",
    "ortho": "ortho",
    "forward": "backward",
}


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
        ) from None


def fft(a, n=None, axis=-1, norm=None, out=None):
    """Compute the one-dimensional discrete Fourier Transform."""
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    return _raw_fft(a, n, axis, False, True, norm, out)


def ifft(a, n=None, axis=-1, norm=None, out=None):
    """Compute the one-dimensional inverse discrete Fourier Transform."""
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    return _raw_fft(a, n, axis, False, False, norm, out=out)


def rfft(a, n=None, axis=-1, norm=None, out=None):
    """Compute the one-dimensional discrete Fourier Transform for real input."""
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    return _raw_fft(a, n, axis, True, True, norm, out=out)


def irfft(a, n=None, axis=-1, norm=None, out=None):
    """Computes the inverse of `rfft`."""
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    return _raw_fft(a, n, axis, True, False, norm, out=out)


def hfft(a, n=None, axis=-1, norm=None, out=None):
    """Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum."""
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    new_norm = _swap_direction(norm)
    return irfft(conjugate(a), n, axis, norm=new_norm, out=out)


def ihfft(a, n=None, axis=-1, norm=None, out=None):
    """Compute the inverse FFT of a signal that has Hermitian symmetry."""
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    new_norm = _swap_direction(norm)
    out = rfft(a, n, axis, norm=new_norm, out=out)
    return conjugate(out, out=out)


def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = True
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = False
    s = list(s)
    if axes is None:
        if not shapeless:
            msg = (
                "`axes` should not be `None` if `s` is not `None` "
                "(Deprecated in NumPy 2.0). In a future version of NumPy, "
                "this will raise an error and `s[i]` will correspond to "
                "the size along the transformed axis specified by "
                "`axes[i]`. To retain current behaviour, pass a sequence "
                "[0, ..., k-1] to `axes` for an array of dimension k."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    if None in s:
        msg = (
            "Passing an array containing `None` values to `s` is "
            "deprecated in NumPy 2.0 and will raise an error in "
            "a future version of NumPy. To use the default behaviour "
            "of the corresponding 1-D transform, pass the value matching "
            "the default for its `n` parameter. To use the default "
            "behaviour for every axis, the `s` argument can be omitted."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
    # use the whole input array along axis `i` if `s[i] == -1`
    s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
    return s, axes


def _raw_fftnd(a, s=None, axes=None, function=fft, norm=None, out=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    itl = list(range(len(axes)))
    itl.reverse()
    for ii in itl:
        a = function(a, n=s[ii], axis=axes[ii], norm=norm, out=out)
    return a


def fftn(a, s=None, axes=None, norm=None, out=None):
    """Compute the N-dimensional discrete Fourier Transform."""
    return _raw_fftnd(a, s, axes, fft, norm, out=out)


def ifftn(a, s=None, axes=None, norm=None, out=None):
    """Compute the N-dimensional inverse discrete Fourier Transform."""
    return _raw_fftnd(a, s, axes, ifft, norm, out=out)


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """Compute the 2-dimensional discrete Fourier Transform."""
    return _raw_fftnd(a, s, axes, fft, norm, out=out)


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """Compute the 2-dimensional inverse discrete Fourier Transform."""
    return _raw_fftnd(a, s, axes, ifft, norm, out=out)


def rfftn(a, s=None, axes=None, norm=None, out=None):
    """Compute the N-dimensional discrete Fourier Transform for real input."""
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    a = rfft(a, s[-1], axes[-1], norm, out=out)
    for ii in range(len(axes) - 2, -1, -1):
        a = fft(a, s[ii], axes[ii], norm, out=out)
    return a


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """Compute the 2-dimensional FFT of a real array."""
    return rfftn(a, s, axes, norm, out=out)


def irfftn(a, s=None, axes=None, norm=None, out=None):
    """Computes the inverse of `rfftn`."""
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes) - 1):
        a = ifft(a, s[ii], axes[ii], norm)
    a = irfft(a, s[-1], axes[-1], norm, out=out)
    return a


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    """Computes the inverse of `rfft2`."""
    return irfftn(a, s, axes, norm, out=out)


integer_types = (int, integer)


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]

    return roll(x, shift, axes)


def ifftshift(x, axes=None):
    """The inverse of `fftshift`. Although identical for even-length `x`, the functions differ
    by one sample for odd-length `x`."""
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, integer_types):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    return roll(x, shift, axes)


def _check_device(device):
    if device not in (None, "cpu"):
        raise ValueError(f'Device not understood. Only "cpu" is allowed, but received: {device}')


def fftfreq(n, d=1.0, device=None):
    """Return the Discrete Fourier Transform sample frequencies."""
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    _check_device(device)
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n - 1) // 2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return results * val


def rfftfreq(n, d=1.0, device=None):
    """Return the Discrete Fourier Transform sample frequencies (for usage with rfft,
    irfft)."""
    if not isinstance(n, integer_types):
        raise ValueError("n should be an integer")
    _check_device(device)
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = arange(0, N, dtype=int)
    return results * val
