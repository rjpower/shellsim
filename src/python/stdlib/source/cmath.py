"""Complex elementary functions composed from the real ``math`` module.

Each function accepts anything ``complex()`` accepts. Results agree with CPython to a few
units in the last place; the branch cuts follow the principal values of ``log`` and ``sqrt``.
"""

import math

__all__ = [
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cos", "cosh", "e", "exp", "inf", "infj",
    "isclose", "isfinite", "isinf", "isnan", "log", "log10", "nan", "nanj", "phase", "pi",
    "polar", "rect", "sin", "sinh", "sqrt", "tan", "tanh", "tau",
]

pi = math.pi
e = math.e
tau = math.tau
inf = math.inf
nan = math.nan
infj = complex(0.0, math.inf)
nanj = complex(0.0, math.nan)

_J = complex(0.0, 1.0)


def phase(value):
    value = complex(value)
    return math.atan2(value.imag, value.real)


def polar(value):
    value = complex(value)
    return abs(value), phase(value)


def rect(radius, angle):
    if math.isinf(radius) and not math.isinf(angle) and not math.isnan(angle):
        if angle == 0.0:
            return complex(radius, angle)
    return complex(radius * math.cos(angle), radius * math.sin(angle))


def sqrt(value):
    value = complex(value)
    if value.real == 0.0 and value.imag == 0.0:
        return complex(0.0, value.imag)
    magnitude = abs(value)
    real = math.sqrt((magnitude + abs(value.real)) / 2)
    imag = abs(value.imag) / (2 * real)
    if value.real < 0:
        real, imag = imag, real
    if value.imag < 0 or (value.imag == 0.0 and math.copysign(1.0, value.imag) < 0):
        imag = -imag
    return complex(real, imag)


def exp(value):
    value = complex(value)
    scale = math.exp(value.real)
    if value.imag == 0.0:
        return complex(scale, value.imag)
    return complex(scale * math.cos(value.imag), scale * math.sin(value.imag))


def log(value, base=None):
    value = complex(value)
    if value.real == 0.0 and value.imag == 0.0:
        raise ValueError("math domain error")
    result = complex(math.log(abs(value)), phase(value))
    if base is None:
        return result
    return result / log(base)


def log10(value):
    return log(value) / math.log(10.0)


def cos(value):
    value = complex(value)
    return complex(
        math.cos(value.real) * math.cosh(value.imag),
        -math.sin(value.real) * math.sinh(value.imag),
    )


def sin(value):
    value = complex(value)
    return complex(
        math.sin(value.real) * math.cosh(value.imag),
        math.cos(value.real) * math.sinh(value.imag),
    )


def tan(value):
    value = complex(value)
    return sin(value) / cos(value)


def cosh(value):
    value = complex(value)
    return complex(
        math.cosh(value.real) * math.cos(value.imag),
        math.sinh(value.real) * math.sin(value.imag),
    )


def sinh(value):
    value = complex(value)
    return complex(
        math.sinh(value.real) * math.cos(value.imag),
        math.cosh(value.real) * math.sin(value.imag),
    )


def tanh(value):
    value = complex(value)
    return sinh(value) / cosh(value)


def asinh(value):
    value = complex(value)
    return log(value + sqrt(value * value + 1))


def acosh(value):
    value = complex(value)
    return log(value + sqrt(value + 1) * sqrt(value - 1))


def atanh(value):
    value = complex(value)
    if value.imag == 0.0 and abs(value.real) == 1.0:
        raise ValueError("math domain error")
    return (log(1 + value) - log(1 - value)) / 2


def asin(value):
    value = complex(value)
    result = asinh(complex(-value.imag, value.real))
    return complex(result.imag, -result.real)


def acos(value):
    value = complex(value)
    result = asin(value)
    return complex(math.pi / 2 - result.real, -result.imag)


def atan(value):
    value = complex(value)
    if value.real == 0.0 and abs(value.imag) == 1.0:
        raise ValueError("math domain error")
    result = atanh(complex(-value.imag, value.real))
    return complex(result.imag, 0.0 - result.real)


def isfinite(value):
    value = complex(value)
    return math.isfinite(value.real) and math.isfinite(value.imag)


def isinf(value):
    value = complex(value)
    return math.isinf(value.real) or math.isinf(value.imag)


def isnan(value):
    value = complex(value)
    return math.isnan(value.real) or math.isnan(value.imag)


def isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0):
    a = complex(a)
    b = complex(b)
    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError("tolerances must be non-negative")
    if a == b:
        return True
    if isinf(a) or isinf(b):
        return False
    difference = abs(a - b)
    return difference <= rel_tol * abs(b) or difference <= rel_tol * abs(a) or difference <= abs_tol
