"""Capability-free complex numbers for the ordinary numeric protocol."""


def _component(value):
    import math
    if not math.isinf(value) and not math.isnan(value) and value == int(value):
        return str(int(value))
    return str(value)


class complex:
    def __init__(self, real=0, imag=0):
        if isinstance(real, complex):
            if imag != 0:
                raise TypeError("complex() second arg can't be a complex number")
            self._real = real.real
            self._imag = real.imag
            return
        self._real = float(real)
        self._imag = float(imag)

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    def conjugate(self):
        return complex(self._real, -self._imag)

    def __bool__(self):
        return self._real != 0 or self._imag != 0

    def __abs__(self):
        import math
        return math.hypot(self._real, self._imag)

    def __add__(self, other):
        other = complex(other)
        return complex(self._real + other.real, self._imag + other.imag)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = complex(other)
        return complex(self._real - other.real, self._imag - other.imag)

    def __rsub__(self, other):
        return complex(other) - self

    def __mul__(self, other):
        other = complex(other)
        return complex(
            self._real * other.real - self._imag * other.imag,
            self._real * other.imag + self._imag * other.real,
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = complex(other)
        denominator = other.real * other.real + other.imag * other.imag
        if denominator == 0:
            raise ZeroDivisionError("complex division by zero")
        return complex(
            (self._real * other.real + self._imag * other.imag) / denominator,
            (self._imag * other.real - self._real * other.imag) / denominator,
        )

    def __rtruediv__(self, other):
        return complex(other) / self

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError("complex exponent must be an integer in shellsim")
        if exponent < 0:
            return complex(1) / (self ** -exponent)
        result = complex(1)
        base = self
        while exponent:
            if exponent % 2:
                result = result * base
            exponent //= 2
            if exponent:
                base = base * base
        return result

    def __eq__(self, other):
        try:
            other = complex(other)
        except (TypeError, ValueError):
            return False
        return self._real == other.real and self._imag == other.imag

    def __neg__(self):
        return complex(-self._real, -self._imag)

    def __pos__(self):
        return self

    def __repr__(self):
        if self._real == 0:
            return _component(self._imag) + "j"
        sign = "+" if self._imag >= 0 else ""
        return "(" + _component(self._real) + sign + _component(self._imag) + "j)"

    def __str__(self):
        return self.__repr__()
