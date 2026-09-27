"""Discrete Fourier transforms.

Every function here is a thin name binding onto the native `_numpy_fft` module, which resolves
NumPy's `n`/`s`, `axis`/`axes`, `norm`, and `out` conventions and runs an O(n log n) transform
(a power-of-two radix-2 Cooley-Tukey transform, and Bluestein's chirp-z transform on top of it
for every other length).
"""

from _numpy_fft import (
    fft,
    fft2,
    fftfreq,
    fftn,
    fftshift,
    hfft,
    ifft,
    ifft2,
    ifftn,
    ifftshift,
    ihfft,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfftfreq,
    rfftn,
)

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
