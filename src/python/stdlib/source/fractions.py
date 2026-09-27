"""Exact rational arithmetic over the VM's metered arbitrary-precision integers.

``Fraction`` accepts integer pairs, fractions, rational or decimal strings, and finite floats or
other objects with an ``as_integer_ratio`` method.
Operations with an int or a Fraction stay exact; with a float or complex they give a float or
complex. Hashes follow Python's numeric hash, so ``hash(Fraction(1, 2)) == hash(0.5)``.
``Decimal`` operands, the ``numbers`` ABCs, and ``__format__`` are not supported.
"""

import math as _math

__all__ = ["Fraction"]

# Python's numeric hash reduces rationals modulo this prime (the language reference's
# "Hashing of numeric types"); a denominator it divides hashes as infinity.
_HASH_MODULUS = 2**61 - 1
_HASH_INFINITY = 314159


def _rational(value):
    """``value`` as a (numerator, denominator) pair if it is an int or Fraction, else None."""
    if isinstance(value, Fraction):
        return value.numerator, value.denominator
    if isinstance(value, int):
        return int(value), 1
    return None


def _string_ratio(text):
    try:
        return _parse_ratio(text.strip())
    except ValueError:
        raise ValueError("Invalid literal for Fraction: %r" % (text,)) from None


def _parse_ratio(text):
    """``text`` as a (numerator, denominator) pair, from ``n/d`` or a decimal with an optional
    exponent. ``int`` reads each digit run, so single underscores may separate digits. Malformed
    input raises ValueError."""
    if "/" in text:
        numerator, denominator = text.split("/")
        if denominator.strip()[:1] in ("+", "-"):
            raise ValueError
        return int(numerator), int(denominator)
    sign = -1 if text[:1] == "-" else 1
    if text[:1] in ("+", "-"):
        text = text[1:]
    mantissa, marker, exponent = text.lower().partition("e")
    whole, _, fractional = mantissa.partition(".")
    if not (whole or fractional):
        raise ValueError
    # int accepts surrounding whitespace, signs and edge underscores, which a Fraction does not.
    for digits in (whole, fractional, exponent.lstrip("+-")):
        if digits and not (digits[0].isdigit() and digits[-1].isdigit()):
            raise ValueError
    numerator = sign * int(whole + fractional)
    scale = len(fractional.replace("_", "")) - (int(exponent) if marker else 0)
    if scale < 0:
        return numerator * 10 ** (-scale), 1
    return numerator, 10**scale


class Fraction:
    def __init__(self, numerator=0, denominator=None):
        if denominator is None:
            ratio = getattr(numerator, "as_integer_ratio", None)
            if isinstance(numerator, str):
                numerator, denominator = _string_ratio(numerator)
            elif _rational(numerator) is not None:
                numerator, denominator = _rational(numerator)
            elif ratio is not None:
                numerator, denominator = ratio()
            else:
                raise TypeError(
                    "argument should be a string or a Rational instance "
                    "or have the as_integer_ratio() method"
                )
        else:
            top = _rational(numerator)
            bottom = _rational(denominator)
            if top is None or bottom is None:
                raise TypeError("both arguments should be Rational instances")
            numerator, denominator = top[0] * bottom[1], top[1] * bottom[0]
        if denominator == 0:
            raise ZeroDivisionError("Fraction(%s, 0)" % numerator)
        divisor = _math.gcd(numerator, denominator)
        if denominator < 0:
            divisor = -divisor
        self._numerator = numerator // divisor
        self._denominator = denominator // divisor

    @classmethod
    def from_float(cls, value):
        if isinstance(value, int):
            return cls(value)
        if not isinstance(value, float):
            raise TypeError(
                "%s.from_float() only takes floats, not %r (%s)"
                % (cls.__name__, value, type(value).__name__)
            )
        return cls(*value.as_integer_ratio())

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def as_integer_ratio(self):
        return self._numerator, self._denominator

    def is_integer(self):
        return self._denominator == 1

    def limit_denominator(self, max_denominator=1000000):
        """The closest Fraction with a denominator of at most ``max_denominator``."""
        if max_denominator < 1:
            raise ValueError("max_denominator should be at least 1")
        if self._denominator <= max_denominator:
            return Fraction(self)
        # Walk the continued fraction; the last convergent and the best semiconvergent within
        # the bound bracket the value, and the nearer one is the answer.
        p0, q0, p1, q1 = 0, 1, 1, 0
        n, d = self._numerator, self._denominator
        while True:
            a = n // d
            q2 = q0 + a * q1
            if q2 > max_denominator:
                break
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            n, d = d, n - a * d
        k = (max_denominator - q0) // q1
        semiconvergent = Fraction(p0 + k * p1, q0 + k * q1)
        convergent = Fraction(p1, q1)
        if abs(convergent - self) <= abs(semiconvergent - self):
            return convergent
        return semiconvergent

    def __repr__(self):
        return "Fraction(%s, %s)" % (self._numerator, self._denominator)

    def __str__(self):
        if self._denominator == 1:
            return str(self._numerator)
        return "%s/%s" % (self._numerator, self._denominator)

    def __hash__(self):
        inverse = pow(self._denominator, _HASH_MODULUS - 2, _HASH_MODULUS)
        if inverse == 0:
            value = _HASH_INFINITY
        else:
            value = abs(self._numerator) % _HASH_MODULUS * inverse % _HASH_MODULUS
        if self._numerator < 0:
            value = -value
        return -2 if value == -1 else value

    def __bool__(self):
        return self._numerator != 0

    def __int__(self):
        return self.__trunc__()

    def __float__(self):
        return self._numerator / self._denominator

    def __trunc__(self):
        if self._numerator < 0:
            return -(-self._numerator // self._denominator)
        return self._numerator // self._denominator

    def __floor__(self):
        return self._numerator // self._denominator

    def __ceil__(self):
        return -(-self._numerator // self._denominator)

    def __round__(self, ndigits=None):
        """Round half to even, to an int, or to a Fraction with ``ndigits`` decimal places."""
        if ndigits is None:
            floor, remainder = divmod(self._numerator, self._denominator)
            if remainder * 2 < self._denominator:
                return floor
            if remainder * 2 > self._denominator:
                return floor + 1
            return floor + floor % 2
        shift = 10 ** abs(ndigits)
        if ndigits > 0:
            return Fraction(round(self * shift), shift)
        return Fraction(round(self / shift) * shift)

    def __neg__(self):
        return Fraction(-self._numerator, self._denominator)

    def __pos__(self):
        return Fraction(self._numerator, self._denominator)

    def __abs__(self):
        return Fraction(abs(self._numerator), self._denominator)

    def __add__(self, other):
        pair = _rational(other)
        if pair is not None:
            n, d = pair
            return Fraction(self._numerator * d + n * self._denominator, self._denominator * d)
        if isinstance(other, (float, complex)):
            return float(self) + other
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        pair = _rational(other)
        if pair is not None:
            n, d = pair
            return Fraction(self._numerator * d - n * self._denominator, self._denominator * d)
        if isinstance(other, (float, complex)):
            return float(self) - other
        return NotImplemented

    def __rsub__(self, other):
        difference = self.__sub__(other)
        return difference if difference is NotImplemented else -difference

    def __mul__(self, other):
        pair = _rational(other)
        if pair is not None:
            return Fraction(self._numerator * pair[0], self._denominator * pair[1])
        if isinstance(other, (float, complex)):
            return float(self) * other
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        pair = _rational(other)
        if pair is not None:
            return Fraction(self._numerator * pair[1], self._denominator * pair[0])
        if isinstance(other, (float, complex)):
            return float(self) / other
        return NotImplemented

    def __rtruediv__(self, other):
        if _rational(other) is not None:
            return Fraction(other) / self
        if isinstance(other, (float, complex)):
            return other / float(self)
        return NotImplemented

    def __floordiv__(self, other):
        pair = _rational(other)
        if pair is not None:
            return (self._numerator * pair[1]) // (self._denominator * pair[0])
        if isinstance(other, float):
            return float(self) // other
        return NotImplemented

    def __rfloordiv__(self, other):
        if _rational(other) is not None:
            return Fraction(other) // self
        if isinstance(other, float):
            return other // float(self)
        return NotImplemented

    def __mod__(self, other):
        pair = _rational(other)
        if pair is not None:
            n, d = pair
            remainder = (self._numerator * d) % (n * self._denominator)
            return Fraction(remainder, self._denominator * d)
        if isinstance(other, float):
            return float(self) % other
        return NotImplemented

    def __rmod__(self, other):
        if _rational(other) is not None:
            return Fraction(other) % self
        if isinstance(other, float):
            return other % float(self)
        return NotImplemented

    def __divmod__(self, other):
        quotient = self.__floordiv__(other)
        if quotient is NotImplemented:
            return NotImplemented
        return quotient, self.__mod__(other)

    def __rdivmod__(self, other):
        if _rational(other) is not None:
            return divmod(Fraction(other), self)
        if isinstance(other, float):
            return divmod(other, float(self))
        return NotImplemented

    def __pow__(self, exponent):
        pair = _rational(exponent)
        if pair is not None and pair[1] == 1:
            power = pair[0]
            if power >= 0:
                return Fraction(self._numerator**power, self._denominator**power)
            return Fraction(self._denominator ** (-power), self._numerator ** (-power))
        if pair is not None:
            return float(self) ** float(exponent)
        if isinstance(exponent, (float, complex)):
            return float(self) ** exponent
        return NotImplemented

    def __rpow__(self, base):
        if _rational(base) is None and not isinstance(base, (float, complex)):
            return NotImplemented
        if self._denominator == 1 and self._numerator >= 0:
            return base**self._numerator
        if _rational(base) is not None:
            return Fraction(base) ** self
        return base ** float(self)

    def __eq__(self, other):
        pair = _rational(other)
        if pair is not None:
            return self._numerator == pair[0] and self._denominator == pair[1]
        if isinstance(other, complex):
            return other.imag == 0 and self == other.real
        if isinstance(other, float):
            if not _math.isfinite(other):
                return False
            return self == Fraction(other)
        return NotImplemented

    def _compare(self, other):
        """The sign of ``self - other`` for a real ``other``, or None if it is not a number.
        Infinities order beyond every Fraction; NaN is unordered, which the caller treats as
        false."""
        if isinstance(other, float):
            if _math.isnan(other):
                return None
            if _math.isinf(other):
                return -1 if other > 0 else 1
            other = Fraction(other)
        pair = _rational(other)
        if pair is None:
            return NotImplemented
        difference = self._numerator * pair[1] - pair[0] * self._denominator
        return (difference > 0) - (difference < 0)

    def __lt__(self, other):
        sign = self._compare(other)
        return sign if sign is NotImplemented else sign is not None and sign < 0

    def __le__(self, other):
        sign = self._compare(other)
        return sign if sign is NotImplemented else sign is not None and sign <= 0

    def __gt__(self, other):
        sign = self._compare(other)
        return sign if sign is NotImplemented else sign is not None and sign > 0

    def __ge__(self, other):
        sign = self._compare(other)
        return sign if sign is NotImplemented else sign is not None and sign >= 0
