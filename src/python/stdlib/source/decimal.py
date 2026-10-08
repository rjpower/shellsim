"""Bounded finite decimal arithmetic over exact, metered Python integers.

This covers the decimal strings, arithmetic, precision, and quantization used by ordinary
data-processing code. Special values and exponent magnitudes above 1,000 are outside this
surface; rejecting them avoids unbounded powers of ten from untrusted input.
"""

ROUND_HALF_EVEN = "ROUND_HALF_EVEN"
ROUND_HALF_UP = "ROUND_HALF_UP"
_MAX_EXPONENT = 1000


class InvalidOperation(ArithmeticError):
    pass


class DivisionByZero(ZeroDivisionError):
    pass


class Context:
    def __init__(self):
        self.prec = 28
        self.rounding = ROUND_HALF_EVEN


_context = Context()


def getcontext():
    return _context


def _power(exponent):
    if not 0 <= exponent <= _MAX_EXPONENT:
        raise InvalidOperation("decimal exponent is outside the supported range")
    return 10**exponent


def _parts(text):
    if not text or len(text) > 4096:
        raise InvalidOperation("invalid decimal")
    text = text.strip()
    sign = -1 if text.startswith("-") else 1
    if text[:1] in ("+", "-"):
        text = text[1:]
    mantissa, marker, exponent_text = text.lower().partition("e")
    try:
        exponent = int(exponent_text) if marker else 0
    except ValueError:
        raise InvalidOperation("invalid decimal") from None
    whole, dot, fractional = mantissa.partition(".")
    digits = whole + fractional
    if not digits or not digits.isdigit() or (dot and not (whole or fractional)):
        raise InvalidOperation("invalid decimal")
    exponent -= len(fractional)
    if abs(exponent) > _MAX_EXPONENT:
        raise InvalidOperation("decimal exponent is outside the supported range")
    return sign * int(digits), exponent


def _rounded_quotient(numerator, denominator, mode):
    if mode not in (ROUND_HALF_EVEN, ROUND_HALF_UP):
        raise ValueError("unsupported decimal rounding mode")
    if denominator == 0:
        raise DivisionByZero("division by zero")
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    quotient, remainder = divmod(abs(numerator), abs(denominator))
    twice = remainder * 2
    if twice > abs(denominator) or (
        twice == abs(denominator)
        and (mode == ROUND_HALF_UP or (mode == ROUND_HALF_EVEN and quotient % 2))
    ):
        quotient += 1
    return sign * quotient


class Decimal:
    def __init__(self, value=0):
        if hasattr(self, "_coefficient"):
            raise TypeError("Decimal cannot be reinitialized")
        if isinstance(value, Decimal):
            coefficient, exponent = value._coefficient, value._exponent
        elif isinstance(value, int):
            coefficient, exponent = value, 0
        elif isinstance(value, str):
            coefficient, exponent = _parts(value)
        else:
            raise TypeError("Decimal requires a string, integer, or Decimal")
        object.__setattr__(self, "_coefficient", coefficient)
        object.__setattr__(self, "_exponent", exponent)

    def __setattr__(self, name, value):
        raise AttributeError("Decimal is immutable")

    @classmethod
    def _make(cls, coefficient, exponent):
        if abs(exponent) > _MAX_EXPONENT:
            raise InvalidOperation("decimal exponent is outside the supported range")
        value = cls(0)
        object.__setattr__(value, "_coefficient", coefficient)
        object.__setattr__(value, "_exponent", exponent)
        return value

    def _plain(self):
        digits = str(abs(self._coefficient))
        exponent = self._exponent
        if exponent >= 0:
            digits += "0" * exponent
        elif -exponent >= len(digits):
            digits = "0." + "0" * (-exponent - len(digits)) + digits
        else:
            digits = digits[:exponent] + "." + digits[exponent:]
        return ("-" if self._coefficient < 0 else "") + digits

    def __str__(self):
        return self._plain()

    def __repr__(self):
        return "Decimal(" + repr(str(self)) + ")"

    def __format__(self, spec):
        if spec in ("", "f"):
            return self._plain()
        if spec.startswith(".") and spec.endswith("f") and spec[1:-1].isdigit():
            places = int(spec[1:-1])
            if places > _MAX_EXPONENT:
                raise ValueError("Decimal format precision is outside the supported range")
            shift = self._exponent + places
            if shift >= 0:
                coefficient = self._coefficient * _power(shift)
            else:
                coefficient = _rounded_quotient(self._coefficient, _power(-shift), _context.rounding)
            return Decimal._make(coefficient, -places)._plain()
        raise ValueError("unsupported Decimal format specification")

    def __bool__(self):
        return self._coefficient != 0

    def is_zero(self):
        return self._coefficient == 0

    def adjusted(self):
        if self._coefficient == 0:
            return 0
        return len(str(abs(self._coefficient))) - 1 + self._exponent

    def __hash__(self):
        from fractions import Fraction

        if self._exponent >= 0:
            return hash(self._coefficient * _power(self._exponent))
        return hash(Fraction(self._coefficient, _power(-self._exponent)))

    def _aligned(self, other):
        if not isinstance(other, Decimal):
            if not isinstance(other, int):
                return NotImplemented
            other = Decimal(other)
        exponent = min(self._exponent, other._exponent)
        left = self._coefficient * _power(self._exponent - exponent)
        right = other._coefficient * _power(other._exponent - exponent)
        return left, right, exponent

    @classmethod
    def _rounded(cls, coefficient, exponent):
        precision = _context.prec
        if not isinstance(precision, int) or not 1 <= precision <= _MAX_EXPONENT:
            raise InvalidOperation("decimal precision is outside the supported range")
        digits = len(str(abs(coefficient)))
        if digits > precision:
            shift = digits - precision
            coefficient = _rounded_quotient(coefficient, _power(shift), _context.rounding)
            exponent += shift
            if len(str(abs(coefficient))) > precision:
                coefficient //= 10
                exponent += 1
        return cls._make(coefficient, exponent)

    def __eq__(self, other):
        aligned = self._aligned(other)
        if aligned is NotImplemented:
            return NotImplemented
        return aligned[0] == aligned[1]

    def __lt__(self, other):
        aligned = self._aligned(other)
        if aligned is NotImplemented:
            return NotImplemented
        return aligned[0] < aligned[1]

    def __le__(self, other):
        if self._aligned(other) is NotImplemented:
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other):
        if self._aligned(other) is NotImplemented:
            return NotImplemented
        return not self <= other

    def __ge__(self, other):
        if self._aligned(other) is NotImplemented:
            return NotImplemented
        return not self < other

    def __neg__(self):
        return Decimal._make(-self._coefficient, self._exponent)

    def __abs__(self):
        return Decimal._make(abs(self._coefficient), self._exponent)

    def __add__(self, other):
        aligned = self._aligned(other)
        if aligned is NotImplemented:
            return NotImplemented
        return Decimal._rounded(aligned[0] + aligned[1], aligned[2])

    __radd__ = __add__

    def __sub__(self, other):
        aligned = self._aligned(other)
        if aligned is NotImplemented:
            return NotImplemented
        return Decimal._rounded(aligned[0] - aligned[1], aligned[2])

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, Decimal):
            if not isinstance(other, int):
                return NotImplemented
            other = Decimal(other)
        return Decimal._rounded(
            self._coefficient * other._coefficient,
            self._exponent + other._exponent,
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Decimal):
            if not isinstance(other, int):
                return NotImplemented
            other = Decimal(other)
        numerator = self._coefficient
        denominator = other._coefficient
        if denominator == 0:
            raise DivisionByZero("division by zero")
        if numerator == 0:
            return Decimal(0)
        exponent = self._exponent - other._exponent
        magnitude = len(str(abs(numerator))) - len(str(abs(denominator))) + exponent
        shift = magnitude - exponent
        if (abs(numerator) < abs(denominator) * _power(shift)) if shift >= 0 else (
            abs(numerator) * _power(-shift) < abs(denominator)
        ):
            magnitude -= 1
        precision = _context.prec
        if not isinstance(precision, int) or not 1 <= precision <= _MAX_EXPONENT:
            raise InvalidOperation("decimal precision is outside the supported range")
        target = magnitude - precision + 1
        shift = exponent - target
        if shift >= 0:
            coefficient = _rounded_quotient(numerator * _power(shift), denominator, _context.rounding)
        else:
            coefficient = _rounded_quotient(numerator, denominator * _power(-shift), _context.rounding)
        return Decimal._make(coefficient, target)

    def __rtruediv__(self, other):
        return Decimal(other) / self

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            return NotImplemented
        if abs(exponent) > _MAX_EXPONENT:
            raise InvalidOperation("decimal power is outside the supported range")
        if exponent < 0:
            return Decimal(1) / (self ** (-exponent))
        return Decimal._rounded(self._coefficient**exponent, self._exponent * exponent)

    def quantize(self, other, rounding=None):
        if not isinstance(other, Decimal):
            raise TypeError("quantize requires a Decimal exponent")
        mode = _context.rounding if rounding is None else rounding
        if mode not in (ROUND_HALF_EVEN, ROUND_HALF_UP):
            raise ValueError("unsupported decimal rounding mode")
        shift = self._exponent - other._exponent
        if shift >= 0:
            coefficient = self._coefficient * _power(shift)
        else:
            coefficient = _rounded_quotient(self._coefficient, _power(-shift), mode)
        if len(str(abs(coefficient))) > _context.prec:
            raise InvalidOperation("quantized result exceeds decimal precision")
        return Decimal._make(coefficient, other._exponent)
