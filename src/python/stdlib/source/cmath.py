"""Deterministic complex elementary functions composed from the math shim."""

import math

pi = math.pi
e = math.e
tau = math.tau
inf = math.inf
nan = math.nan


def phase(value):
    value = complex(value)
    return math.atan2(value.imag, value.real)


def polar(value):
    value = complex(value)
    return abs(value), phase(value)


def rect(radius, angle):
    return complex(radius * math.cos(angle), radius * math.sin(angle))


def sqrt(value):
    value = complex(value)
    magnitude = abs(value)
    real = math.sqrt((magnitude + value.real) / 2)
    imag = math.sqrt((magnitude - value.real) / 2)
    if value.imag < 0:
        imag = -imag
    return complex(real, imag)


def exp(value):
    value = complex(value)
    scale = math.exp(value.real)
    return complex(scale * math.cos(value.imag), scale * math.sin(value.imag))


def log(value, base=None):
    value = complex(value)
    result = complex(math.log(abs(value)), phase(value))
    if base is None:
        return result
    return result / log(base)


def cos(value):
    value = complex(value)
    positive = math.exp(value.imag)
    negative = math.exp(-value.imag)
    cosh = (positive + negative) / 2
    sinh = (positive - negative) / 2
    return complex(math.cos(value.real) * cosh, -math.sin(value.real) * sinh)


def sin(value):
    value = complex(value)
    positive = math.exp(value.imag)
    negative = math.exp(-value.imag)
    cosh = (positive + negative) / 2
    sinh = (positive - negative) / 2
    return complex(math.sin(value.real) * cosh, math.cos(value.real) * sinh)
