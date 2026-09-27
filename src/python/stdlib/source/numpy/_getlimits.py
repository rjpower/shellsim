"""``finfo`` and ``iinfo``: the numeric limits of a dtype.

Every value here follows directly from the IEEE 754 layout of a float dtype (mantissa and
exponent bit counts) or from the bit width of an integer dtype, computed with the standard
formulas rather than a table of literals copied from any implementation.
"""

import math

import numpy as np

__all__ = ["finfo", "iinfo"]

# Mantissa and exponent bit counts (excluding the sign bit and, for the mantissa, the implicit
# leading one) for the three floating dtypes shellsim models.
_LAYOUT = {
    "float16": (10, 5),
    "float32": (23, 8),
    "float64": (52, 11),
}
# A complex dtype's numeric limits are those of its real and imaginary components.
_COMPLEX_TO_FLOAT = {"complex64": "float32", "complex128": "float64"}


def _resolve_dtype(dtype_like):
    """Resolve any of `finfo`/`iinfo`'s accepted spellings to a ``numpy.dtype``.

    NumPy accepts a dtype, a scalar type, a dtype-spec string, or a value whose *type* names a
    dtype (``finfo(3)`` uses ``dtype(int)``, matching NumPy).
    """
    try:
        return np.dtype(dtype_like)
    except TypeError:
        return np.dtype(type(dtype_like))


class finfo:
    """Machine limits for a floating or complex floating dtype.

    Example
    -------
    >>> np.finfo(np.float32).eps
    1.1920929e-07
    """

    def __init__(self, dtype):
        resolved = _resolve_dtype(dtype)
        float_name = _COMPLEX_TO_FLOAT.get(resolved.name, resolved.name)
        if float_name not in _LAYOUT:
            raise ValueError(f"data type {resolved!r} not compatible with finfo")
        nmant, nexp = _LAYOUT[float_name]
        self.dtype = np.dtype(float_name)
        self.bits = self.dtype.itemsize * 8
        precision_bits = nmant + 1
        self.nmant = nmant
        self.nexp = nexp
        self.iexp = nexp
        self.machep = -nmant
        self.negep = -(nmant + 1)
        self.maxexp = 2 ** (nexp - 1)
        self.minexp = -self.maxexp + 2
        self.eps = self.dtype.type(2.0**self.machep)
        self.epsneg = self.dtype.type(2.0**self.negep)
        self.max = self.dtype.type((2.0 - 2.0 ** (1 - precision_bits)) * 2.0 ** (self.maxexp - 1))
        self.min = self.dtype.type(-self.max)
        self.smallest_normal = self.dtype.type(2.0**self.minexp)
        self.tiny = self.smallest_normal
        self.smallest_subnormal = self.dtype.type(2.0 ** (self.minexp - nmant))
        self.precision = int(nmant * math.log10(2))
        self.resolution = self.dtype.type(10.0**-self.precision)

    def __repr__(self):
        return f"finfo(resolution={self.resolution!r}, min={self.min!r}, max={self.max!r}, dtype={self.dtype})"


class iinfo:
    """Machine limits for an integer dtype.

    Example
    -------
    >>> np.iinfo(np.int8).max
    127
    """

    def __init__(self, dtype):
        resolved = _resolve_dtype(dtype)
        if resolved.kind not in "iu":
            raise ValueError(f"Invalid integer data type {resolved.char!r}.")
        self.dtype = resolved
        self.bits = resolved.itemsize * 8
        self.kind = resolved.kind
        if resolved.kind == "u":
            self.min = 0
            self.max = 2**self.bits - 1
        else:
            self.min = -(2 ** (self.bits - 1))
            self.max = 2 ** (self.bits - 1) - 1

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"
