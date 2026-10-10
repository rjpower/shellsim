"""Call the independent provider through upstream ctypes and its Fortran ABI."""

import ctypes as c

library = c.CDLL("/usr/local/lib/libopenblas.so")
integer = c.c_int
real = c.c_double
pint = c.POINTER(integer)
pdouble = c.POINTER(real)

library.dgemm_.argtypes = [
    c.c_char_p,
    c.c_char_p,
    pint,
    pint,
    pint,
    pdouble,
    pdouble,
    pint,
    pdouble,
    pint,
    pdouble,
    pdouble,
    pint,
]
library.dgemm_.restype = integer
n, one = integer(2), integer(1)
a = (real * 4)(1, 3, 2, 4)
b = (real * 4)(5, 7, 6, 8)
result = (real * 4)()
alpha, beta = real(1), real(0)
assert (
    library.dgemm_(
        b"N",
        b"N",
        c.byref(n),
        c.byref(n),
        c.byref(n),
        c.byref(alpha),
        a,
        c.byref(n),
        b,
        c.byref(n),
        c.byref(beta),
        result,
        c.byref(n),
    )
    == 0
)
assert list(result) == [19, 43, 22, 50]

library.dgesv_.argtypes = [pint, pint, pdouble, pint, pint, pdouble, pint, pint]
library.dgesv_.restype = integer
system = (real * 4)(3, 1, 1, 2)
rhs = (real * 2)(9, 8)
pivots = (integer * 2)()
info = integer()
assert library.dgesv_(c.byref(n), c.byref(one), system, c.byref(n), pivots, rhs, c.byref(n), c.byref(info)) == 0
assert info.value == 0 and abs(rhs[0] - 2) < 1e-12 and abs(rhs[1] - 3) < 1e-12
invalid = integer(-1)
library.dgesv_(c.byref(invalid), c.byref(one), system, c.byref(n), pivots, rhs, c.byref(n), c.byref(info))
assert info.value == -1

library.sdot_.argtypes = [pint, c.POINTER(c.c_float), pint, c.POINTER(c.c_float), pint]
library.sdot_.restype = c.c_float
assert library.sdot_(c.byref(n), (c.c_float * 2)(1, 2), c.byref(one), (c.c_float * 2)(3, 4), c.byref(one)) == 11


class Complex(c.Structure):
    _fields_ = [("real", real), ("imag", real)]


library.zdotc_.argtypes = [c.POINTER(Complex), pint, c.POINTER(Complex), pint, c.POINTER(Complex), pint]
library.zdotc_.restype = None
z = Complex()
x = (Complex * 2)(Complex(1, 2), Complex(3, -1))
y = (Complex * 2)(Complex(2, -1), Complex(-1, 4))
library.zdotc_(c.byref(z), c.byref(n), x, c.byref(one), y, c.byref(one))
assert (z.real, z.imag) == (-7, 6)
print("shared OpenBLAS numerical and Fortran ABI passed")
