"""``np.finfo`` and ``np.iinfo``, following ``numpy/_core/getlimits.py``.

NumPy reads floating-point constants from the C library at import; the IEEE formats shellsim
supports have fixed values, so they are tabulated here. Attributes are scalars of the
described type, as in NumPy.
"""

import math

from _numpy import dtype as _dtype

# Per IEEE format: eps, max, smallest normal, smallest subnormal, precision, maxexp, minexp,
# and nmant, from NumPy 2.5 on x86-64.
_CONSTANTS = {
    "float16": (2.0**-10, 65504.0, 2.0**-14, 2.0**-24, 3, 16, -14, 10),
    "float32": (2.0**-23, 3.4028234663852886e38, 2.0**-126, 2.0**-149, 6, 128, -126, 23),
    "float64": (2.0**-52, 1.7976931348623157e308, 2.0**-1022, 5e-324, 15, 1024, -1022, 52),
}
_REPR_FORMATS = {"float16": "%12.5e", "float32": "%15.7e", "float64": "%24.16e"}


def _as_dtype(value):
    try:
        return _dtype(value)
    except TypeError:
        return _dtype(type(value))


class finfo:
    """Machine limits for floating-point types.

    NumPy caches one instance per type in ``__new__``; shellsim classes do not yet run a
    user-defined ``__new__``, so each call builds a new, equal instance.
    """

    def __init__(self, dtype):
        if dtype is None:
            raise TypeError("dtype must not be None")
        dtype = _as_dtype(dtype)
        if dtype.kind == "c":
            real = _dtype(f"f{dtype.itemsize // 2}")
        elif dtype.kind == "f":
            real = dtype
        else:
            raise ValueError(f"data type {dtype!r} not compatible with finfo")
        self._init(real)

    def _init(self, dtype):
        eps, largest, normal, subnormal, precision, maxexp, minexp, nmant = _CONSTANTS[
            dtype.name
        ]
        scalar = dtype.type
        self.dtype = dtype
        self.bits = dtype.itemsize * 8
        self.eps = scalar(eps)
        self.epsneg = scalar(eps / 2)
        self.max = scalar(largest)
        self.min = scalar(-largest)
        self.smallest_normal = scalar(normal)
        self.tiny = self.smallest_normal
        self.smallest_subnormal = scalar(subnormal)
        self.precision = precision
        self.resolution = scalar(10) ** (-precision)
        self.maxexp = maxexp
        self.minexp = minexp
        self.nmant = nmant
        self.machep = int(math.log2(eps))
        self.negep = int(math.log2(eps / 2))
        self.nexp = math.ceil(math.log2(maxexp - minexp + 2))
        self.iexp = math.ceil(math.log2(maxexp - minexp))

    def __str__(self):
        return (
            f"Machine parameters for {self.dtype}\n"
            "---------------------------------------------------------------\n"
            f"precision = {self.precision}   resolution = {self.resolution}\n"
            f"machep = {self.machep}   eps =        {self.eps}\n"
            f"negep =  {self.negep}   epsneg =     {self.epsneg}\n"
            f"minexp = {self.minexp}   tiny =       {self.tiny}\n"
            f"maxexp = {self.maxexp}   max =        {self.max}\n"
            f"nexp =   {self.nexp}   min =        -max\n"
            f"smallest_normal = {self.smallest_normal}   "
            f"smallest_subnormal = {self.smallest_subnormal}\n"
            "---------------------------------------------------------------\n"
        )

    def __repr__(self):
        fmt = _REPR_FORMATS[self.dtype.name]
        max_str = (fmt % float(self.max)).strip()
        min_str = (fmt % float(self.min)).strip()
        return (
            f"finfo(resolution={self.resolution}, min={min_str}, max={max_str}, "
            f"dtype={self.dtype})"
        )


class iinfo:
    """Machine limits for integer types."""

    def __init__(self, int_type):
        self.dtype = _as_dtype(int_type)
        self.kind = self.dtype.kind
        self.bits = self.dtype.itemsize * 8
        self.key = f"{self.kind}{self.bits}"
        if self.kind not in "iu":
            raise ValueError(f"Invalid integer data type {self.kind!r}.")

    @property
    def min(self):
        if self.kind == "u":
            return 0
        return -(1 << (self.bits - 1))

    @property
    def max(self):
        if self.kind == "u":
            return (1 << self.bits) - 1
        return (1 << (self.bits - 1)) - 1

    def __str__(self):
        return (
            f"Machine parameters for {self.dtype}\n"
            "---------------------------------------------------------------\n"
            f"min = {self.min}\n"
            f"max = {self.max}\n"
            "---------------------------------------------------------------\n"
        )

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"
