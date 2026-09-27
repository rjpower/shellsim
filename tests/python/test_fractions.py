"""Portable ``fractions`` semantics, checked against CPython 3.14 by the source-suite harness."""

import math
from fractions import Fraction

INF = float("inf")
NAN = float("nan")

# Literal text and the (numerator, denominator) it denotes.
STRINGS = [
    (" -12 ", (-12, 1)),
    ("3/6", (1, 2)),
    ("1_000 / 2", (500, 1)),
    ("-2/4", (-1, 2)),
    (" -1.25e-2 ", (-1, 80)),
    (".125", (1, 8)),
    ("5.", (5, 1)),
    ("1.5E3", (1500, 1)),
    ("1_0.5_0e1", (105, 1)),
]

MALFORMED = [
    "",
    " ",
    "/",
    "1/",
    "/2",
    "1/-2",
    "1/2/3",
    "1__2",
    "_1",
    "1_",
    "1._5",
    "1.2.3",
    "1e",
    "1e 5",
    "- 1",
    "nan",
    "inf",
    "0x10",
]


def _assert_raises(call, error, message=None):
    try:
        call()
    except error as caught:
        assert message is None or str(caught) == message, str(caught)
    else:
        raise AssertionError(f"{error.__name__} was not raised")


def test_strings_parse_as_exact_ratios():
    for text, expected in STRINGS:
        assert Fraction(text).as_integer_ratio() == expected, text


def test_construction_normalizes_sign_and_common_factors():
    assert Fraction() == 0 and Fraction(7) == 7
    assert repr(Fraction(2, -4)) == "Fraction(-1, 2)" and str(Fraction(2, -4)) == "-1/2"
    assert str(Fraction(6, 3)) == "2"
    assert Fraction(Fraction(1, 3), Fraction(2, 3)) == Fraction(1, 2)
    assert Fraction(True).as_integer_ratio() == (1, 1)
    value = Fraction(3, 6)
    assert (value.numerator, value.denominator) == (1, 2)
    assert Fraction(value) is not value and Fraction(value) == value
    assert Fraction(10).is_integer() and not value.is_integer()


def test_floats_and_ratio_objects_convert_exactly():
    assert Fraction(0.1).as_integer_ratio() == (3602879701896397, 36028797018963968)
    assert Fraction(0.1) != Fraction(1, 10)
    assert Fraction(-2.5) == Fraction(-5, 2)
    assert Fraction(1e100).denominator == 1
    assert Fraction(5e-324).denominator == 2**1074
    assert Fraction.from_float(0.25) == Fraction(1, 4)
    assert Fraction.from_float(3) == 3

    class Ratio:
        def as_integer_ratio(self):
            return 3, 4

    assert Fraction(Ratio()) == Fraction(3, 4)


def test_invalid_construction_raises_cpython_errors():
    for text in MALFORMED:
        _assert_raises(lambda text=text: Fraction(text), ValueError, f"Invalid literal for Fraction: {text!r}")
    _assert_raises(lambda: Fraction(1, 0), ZeroDivisionError, "Fraction(1, 0)")
    _assert_raises(lambda: Fraction(NAN), ValueError, "cannot convert NaN to integer ratio")
    _assert_raises(lambda: Fraction(INF), OverflowError, "cannot convert Infinity to integer ratio")
    _assert_raises(lambda: Fraction(1.5, 2), TypeError, "both arguments should be Rational instances")
    message = "argument should be a string or a Rational instance or have the as_integer_ratio() method"
    _assert_raises(lambda: Fraction(1j), TypeError, message)
    _assert_raises(lambda: Fraction([1]), TypeError, message)
    _assert_raises(
        lambda: Fraction.from_float("1"), TypeError, "Fraction.from_float() only takes floats, not '1' (str)"
    )

    def assign():
        Fraction(1, 2).numerator = 3

    _assert_raises(assign, AttributeError)


def test_arithmetic_with_ints_and_fractions_is_exact():
    third, sixth = Fraction(1, 3), Fraction(1, 6)
    assert third + sixth == Fraction(1, 2) and 1 + third == Fraction(4, 3)
    assert Fraction(2, 3) - sixth == Fraction(1, 2) and 1 - third == Fraction(2, 3)
    assert 3 * third == 1 and type(3 * third) is Fraction
    assert third / 2 == sixth and 2 / third == 6
    assert Fraction(-7, 3) // 2 == -2 and type(Fraction(-7, 3) // 2) is int
    assert Fraction(-7, 3) % 2 == Fraction(5, 3)
    assert divmod(Fraction(7, 2), Fraction(2, 3)) == (5, Fraction(1, 6))
    assert divmod(7, Fraction(2)) == (3, 1)
    assert Fraction(2, 3) ** -2 == Fraction(9, 4) and Fraction(2, 3) ** 0 == 1
    assert -third == Fraction(-1, 3) and +third == third and abs(-third) == third
    assert sum([third, third, third]) == 1
    _assert_raises(lambda: third / 0, ZeroDivisionError)
    _assert_raises(lambda: third % Fraction(0), ZeroDivisionError)


def test_powers_keep_exact_results_when_the_exponent_is_an_integer():
    assert repr(2 ** Fraction(3)) == "8"
    assert repr((-2) ** Fraction(-2)) == "Fraction(1, 4)"
    assert repr(Fraction(1, 4) ** Fraction(1, 2)) == "0.5"
    assert repr(2 ** Fraction(1, 2)) == "1.4142135623730951"
    assert repr(2.0 ** Fraction(3)) == "8.0"
    assert repr(Fraction(-4) ** Fraction(1, 2)) == repr((-4.0) ** 0.5)
    _assert_raises(lambda: Fraction(0) ** -1, ZeroDivisionError)


def test_floats_and_complex_numbers_give_inexact_results():
    half = Fraction(1, 2)
    assert half + 0.25 == 0.75 and type(half + 0.25) is float
    assert 0.25 * half == 0.125 and 1.0 / half == 2.0
    assert half + 1j == complex(0.5, 1)
    assert Fraction(7, 2) % 1.5 == 0.5 and Fraction(7, 2) // 1.5 == 2.0
    assert half**2.0 == 0.25
    assert float(Fraction(1, 3)) == 1 / 3 and int(Fraction(-7, 3)) == -2


def test_comparisons_order_fractions_with_ints_and_floats():
    third = Fraction(1, 3)
    assert third < 0.5 and 0.5 > third and third <= Fraction(2, 6) and third >= 0
    assert Fraction(1, 2) == 0.5 and 0.5 == Fraction(1, 2) and Fraction(1, 2) == complex(0.5, 0)
    assert Fraction(1, 2) != 0.5 + 1j and Fraction(2) == 2
    assert Fraction(10**1000) < INF and Fraction(-(10**1000)) > -INF
    assert not (third < NAN or third <= NAN or third > NAN or third >= NAN or third == NAN)
    assert max(Fraction(1, 2), 0.4) == Fraction(1, 2)
    assert not Fraction(0) and bool(third)


def test_unsupported_operands_raise_type_error():
    half = Fraction(1, 2)
    _assert_raises(lambda: half + "x", TypeError, "unsupported operand type(s) for +: 'Fraction' and 'str'")
    _assert_raises(lambda: "x" - half, TypeError, "unsupported operand type(s) for -: 'str' and 'Fraction'")
    _assert_raises(lambda: half < "x", TypeError, "'<' not supported between instances of 'Fraction' and 'str'")
    _assert_raises(lambda: half < 1j, TypeError, "'<' not supported between instances of 'Fraction' and 'complex'")
    _assert_raises(lambda: half // 1j, TypeError)


def test_rounding_uses_half_even_and_the_rounding_protocol():
    assert [round(Fraction(n, 2)) for n in (-5, -3, -1, 1, 3, 5, 7)] == [-2, -2, 0, 0, 2, 2, 4]
    assert round(Fraction(1, 3)) == 0 and type(round(Fraction(7, 2))) is int
    assert round(Fraction(25, 100), 1) == Fraction(1, 5) and type(round(Fraction(1, 3), 1)) is Fraction
    assert round(Fraction(1234), -2) == 1200 and round(Fraction(1250), -2) == 1200
    assert (math.floor(Fraction(-7, 2)), math.ceil(Fraction(-7, 2)), math.trunc(Fraction(-7, 2))) == (-4, -3, -3)
    assert (math.floor(Fraction(7, 2)), math.ceil(Fraction(7, 2)), math.trunc(Fraction(7, 2))) == (3, 4, 3)
    assert math.sqrt(Fraction(1, 4)) == 0.5 and math.isclose(Fraction(1, 3), 1 / 3)


def test_fractions_and_equal_numbers_are_the_same_container_element():
    half = Fraction(1, 2)
    assert [half, Fraction(2)] == [0.5, 2] and (Fraction(2, 4),) == (half,)
    assert half in [0.5] and 0.5 in {half} and Fraction(4, 8) in {0.5: "half"}
    assert {half: "a", 0.5: "b"} == {half: "b"} and len({half, Fraction(2, 4), 0.5}) == 1
    assert sorted([Fraction(1, 2), 0.25, 1, Fraction(-1, 3)]) == [Fraction(-1, 3), 0.25, half, 1]


def test_hash_agrees_with_equal_numbers():
    assert hash(Fraction(1, 2)) == hash(0.5)
    assert hash(Fraction(3)) == hash(3) and hash(Fraction(-1)) == hash(-1) == -2
    assert hash(Fraction(-5, 4)) == hash(-1.25)
    assert hash(Fraction(1, 3)) == hash(Fraction(2, 6))
    assert hash(Fraction(1, 2**61 - 1)) == hash(INF)
    assert hash(Fraction(10**30, 7)) == hash(Fraction(10**30, 7))


def test_limit_denominator_finds_the_closest_bounded_fraction():
    assert Fraction("3.1415926535897932").limit_denominator(1000) == Fraction(355, 113)
    assert Fraction(0.1).limit_denominator() == Fraction(1, 10)
    assert Fraction(-3, 7).limit_denominator(2) == Fraction(-1, 2)
    assert Fraction(1, 3).limit_denominator(10) == Fraction(1, 3)
    assert Fraction(5, 7).limit_denominator(1) == 1
    _assert_raises(lambda: Fraction(1, 2).limit_denominator(0), ValueError, "max_denominator should be at least 1")
