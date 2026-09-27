# Portable CPython-style probes for common builtin and container behavior.


def test_int_accepts_explicit_bases():
    assert int("ff", 16) == 255
    assert int("0b101", 0) == 5
    assert int("-1_000") == -1000


def test_string_join_replace_and_format():
    assert ",".join(["a", "b", "c"]) == "a,b,c"
    assert "banana".replace("a", "o", 2) == "bonona"
    assert "{}:{}:{name}:{{ok}}".format("x", 2, name="n") == "x:2:n:{ok}"
    assert "{1}:{0}".format("x", 2) == "2:x"


def test_list_reverse_count_and_index():
    values = [1, 2, 1, 3]
    values.reverse()
    assert values == [3, 1, 2, 1]
    assert values.count(1) == 2
    assert values.index(1, 2) == 3


def test_dict_update_accepts_mappings_pairs_and_keywords():
    value = {"a": 1}
    value.update({"b": 2})
    value.update([("c", 3)], d=4)
    assert value == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_dict_pop_supports_defaults():
    value = {"a": 1}
    assert value.pop("a") == 1
    assert value.pop("missing", 3) == 3
    assert value == {}


def test_set_union_accepts_multiple_iterables():
    assert {1, 2}.union({2, 3}, [4]) == {1, 2, 3, 4}


def test_map_accepts_multiple_iterables():
    assert list(map(lambda left, right: left + right, [1, 2], [10, 20])) == [11, 22]


def test_filter_accepts_callables_and_none():
    assert list(filter(lambda value: value > 1, [0, 1, 2, 3])) == [2, 3]
    assert list(filter(None, [0, 1, "", "x"])) == [1, "x"]


def test_getattr_and_hasattr_share_attribute_lookup():
    class Item:
        value = 3

    assert getattr(Item, "value") == 3  # noqa: B009 - exercise the getattr builtin
    assert getattr(Item(), "missing", 4) == 4
    assert hasattr(Item, "value")
    assert not hasattr(Item, "missing")


def test_reversed_returns_an_iterator():
    iterator = reversed((1, 2, 3))
    assert next(iterator) == 3
    assert list(iterator) == [2, 1]


def test_invalid_inputs_raise_python_exceptions():
    try:
        int("10", 1)
        raise AssertionError("int() accepted an invalid base")
    except ValueError:
        pass

    try:
        "".join([1])
        raise AssertionError("str.join() accepted a non-string item")
    except TypeError:
        pass

    try:
        [1].index(2)
        raise AssertionError("list.index() found a missing item")
    except ValueError:
        pass

    try:
        {}.pop("missing")
        raise AssertionError("dict.pop() accepted a missing key")
    except KeyError:
        pass


def _assert_raises(call, error, message):
    try:
        call()
    except error as caught:
        assert str(caught) == message, str(caught)
    else:
        raise AssertionError(f"{message!r} was not raised")


def test_min_and_max_accept_key_and_default():
    assert min([3, 1, 2], key=lambda value: -value) == 3
    assert max(3, 1, 2, key=lambda value: -value) == 1
    assert max([], default=5) == 5
    assert min([], key=len, default="x") == "x"
    assert min([2, 1], default=None) == 1
    assert min(range(1, 5), key=None) == 1
    # Of equal keys, both functions keep the first item.
    pairs = [(1, "a"), (1, "b")]
    assert min(pairs, key=lambda pair: pair[0]) == (1, "a")
    assert max(pairs, key=lambda pair: pair[0]) == (1, "a")
    _assert_raises(lambda: min([]), ValueError, "min() iterable argument is empty")
    _assert_raises(lambda: max(), TypeError, "max expected at least 1 argument, got 0")
    _assert_raises(
        lambda: min(1, 2, default=0),
        TypeError,
        "Cannot specify a default for min() with multiple positional arguments",
    )
    _assert_raises(lambda: min([1], bogus=1), TypeError, "min() got an unexpected keyword argument 'bogus'")
    _assert_raises(lambda: max([1, "a"]), TypeError, "'>' not supported between instances of 'str' and 'int'")


def test_setattr_stores_through_descriptors():
    class Box:
        pass

    class Doubler:
        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, value):
            self._value = value * 2

    box = Box()
    box.size = 3
    assert box.size == 3
    doubler = Doubler()
    doubler.value = 4
    assert doubler.value == 8
    _assert_raises(lambda: setattr(box, 1, 2), TypeError, "attribute name must be string, not 'int'")
    _assert_raises(lambda: setattr(box, "size"), TypeError, "setattr expected 3 arguments, got 2")


def test_numeric_conversions_use_dunder_methods_in_cpython_order():
    class Indexable:
        def __index__(self):
            return 7

    class Floaty:
        def __float__(self):
            return 2.5

    class Complexish:
        def __complex__(self):
            return 1 + 2j

    class Integral:
        def __int__(self):
            return 4

        def __index__(self):
            return 9

    class OnlyInt:
        def __int__(self):
            return 3

    class BadInt:
        def __int__(self):
            return 1.5

    class BadFloat:
        def __float__(self):
            return 1

    class BadComplex:
        def __complex__(self):
            return 1

    # int() prefers __int__ to __index__; float() and complex() fall back to __index__.
    assert int(Indexable()) == 7
    assert int(Integral()) == 4
    assert float(Indexable()) == 7.0
    assert float(Floaty()) == 2.5
    assert complex(Complexish()) == 1 + 2j
    assert complex(Floaty()) == 2.5 + 0j
    assert complex(Indexable(), Floaty()) == 7 + 2.5j
    _assert_raises(lambda: int(BadInt()), TypeError, "__int__ returned non-int (type float)")
    _assert_raises(lambda: float(BadFloat()), TypeError, "BadFloat.__float__ returned non-float (type int)")
    _assert_raises(lambda: complex(BadComplex()), TypeError, "__complex__ returned non-complex (type int)")
    _assert_raises(
        lambda: int(None),
        TypeError,
        "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'",
    )
    _assert_raises(
        lambda: float(OnlyInt()),
        TypeError,
        "float() argument must be a string or a real number, not 'OnlyInt'",
    )
    _assert_raises(
        lambda: float(object()),
        TypeError,
        "float() argument must be a string or a real number, not 'object'",
    )


def test_string_affix_tests_accept_bounds_and_tuples():
    assert "abc".startswith(("x", "b"), 1)
    assert "abc".startswith("b", 1, 2)
    assert "abc".startswith("", 3)
    assert not "abc".startswith("", 4)
    assert "abc".endswith("b", None, -1)
    assert "abc".removeprefix("ab") == "c"
    assert "abc".removesuffix("x") == "abc"


def test_string_case_transforms_use_unicode_title_case():
    assert "hello wORLD 3rd".title() == "Hello World 3Rd"
    assert "\u01c6a \ufb01x \u00df".title() == "\u01c5a Fix Ss"
    assert "\u03a3\u0391\u03a3 x".capitalize() == "\u03a3\u03b1\u03c2 x"
    assert "aB\u00df".swapcase() == "AbSS"
    assert "Ab Cd".istitle() and not "AB".istitle()


def test_string_numeric_and_space_predicates_follow_python():
    assert "\u0663".isdecimal() and not "\u00b2".isdecimal()
    assert "\u00b2".isdigit() and not "\u00bd".isdigit()
    assert "\u00bd\u4e00".isnumeric()
    assert "\x1c \u3000".isspace() and not "".isspace()
    assert "abc".isascii() and not "\u00e9".isascii()


def test_string_expandtabs_and_translate():
    assert "ab\tc\n\tx".expandtabs() == "ab      c\n        x"
    assert "a\tb".expandtabs(tabsize=3) == "a  b"
    assert "abc".translate({97: "xy", 98: None, 99: 66}) == "xyB"
    assert "abc".translate([None] * 98 + ["X"]) == "Xc"


def test_ascii_escapes_non_ascii_characters():
    assert ascii("caf\u00e9 \u4e00 \U0001f600") == "'caf\\xe9 \\u4e00 \\U0001f600'"


def test_float_hex_matches_cpython():
    cases = [
        (1.0, "0x1.0000000000000p+0"),
        (-3.5, "-0x1.c000000000000p+1"),
        (0.1, "0x1.999999999999ap-4"),
        (1e300, "0x1.7e43c8800759cp+996"),
        (5e-324, "0x0.0000000000001p-1022"),
        (-0.0, "-0x0.0p+0"),
        (float("-inf"), "-inf"),
        (float("nan"), "nan"),
    ]
    for value, text in cases:
        assert value.hex() == text


def test_id_matches_identity():
    first = []
    second = []
    assert id(first) == id(first) and id(first) != id(second)
    assert id(5) == id(5) and id(5) != id(6)
    assert f"{id(object()):#x}".startswith("0x")


def test_sys_maxsize_is_64_bit():
    import sys

    assert sys.maxsize == 2**63 - 1


FORMAT_CASES = [
    (5, " 3d", "  5"),
    (5, "*^9", "****5****"),
    (5, "=+6", "+    5"),
    (5, "<05", "50000"),
    (-5, "0^8", "000-5000"),
    ("ab", "05", "ab000"),
    ("abc", "\u00e9^7.2", "\u00e9\u00e9ab\u00e9\u00e9\u00e9"),
    (1234567, "_", "1_234_567"),
    (255, "#_b", "0b1111_1111"),
    (1234.5, "012,.1f", "00,001,234.5"),
    (1234.5, "*=12,.1f", "*****1,234.5"),
    (1234.56789, ",.5_f", "1,234.567_89"),
    (-0.001, "z.1f", "0.0"),
    (float("nan"), "F", "NAN"),
    (float("nan"), "+010.2f", "+000000nan"),
    (65, "5c", "    A"),
    (1.0, "#.0f", "1."),
    (1e20, "#.3g", "1.00e+20"),
    (1e16, "#", "1.e+16"),
    (True, "5", "    1"),
    (123456789, "e", "1.234568e+08"),
]

FORMAT_ERRORS = [
    (5, ".2", ValueError, "Precision not allowed in integer format specifier"),
    (5, ",x", ValueError, "Cannot specify ',' with 'x'."),
    (5, ",_", ValueError, "Cannot specify both ',' and '_'."),
    (5, "5.2ff", ValueError, "Invalid format specifier '5.2ff' for object of type 'int'"),
    (65, "+c", ValueError, "Sign not allowed with integer format specifier 'c'"),
    (-1, "c", OverflowError, "%c arg not in range(0x110000)"),
    (1.5, "d", ValueError, "Unknown format code 'd' for object of type 'float'"),
    ("a", "=5", ValueError, "'=' alignment not allowed in string format specifier"),
    ("a", "+", ValueError, "Sign not allowed in string format specifier"),
    (True, "s", ValueError, "Unknown format code 's' for object of type 'bool'"),
    (None, "5", TypeError, "unsupported format string passed to NoneType.__format__"),
]


def test_format_specifications_match_cpython():
    for value, spec, expected in FORMAT_CASES:
        assert (value, spec, format(value, spec)) == (value, spec, expected)
    for value, spec, kind, message in FORMAT_ERRORS:
        try:
            format(value, spec)
        except Exception as error:
            assert (spec, type(error), str(error)) == (spec, kind, message)
        else:
            raise AssertionError(f"format({value!r}, {spec!r}) did not raise")


def test_percent_formatting_matches_cpython():
    nan = float("nan")
    assert "%.2f|%5.1f|%E|%g|%#g" % (nan, -float("inf"), nan, 1e-5, 1.0) == ("nan| -inf|NAN|1e-05|1.00000")
    assert "%e|%.0e|%#.0f" % (1e10, 12345.0, 1.0) == "1.000000e+10|1e+04|1."
    assert "%#x|%#08X|%.4x|%#.4o|%x" % (0, -255, 255, 8, -255) == "0x0|-0X000FF|00ff|0o0010|-ff"
    assert "%d|%+.3d|% 05d|%-6d|" % (3.7, 5, 5, 3) == "3|+005| 0005|3     |"
    assert "%010f|%-6c|%10.3s" % (nan, 65, "abcdef") == "0000000nan|A     |       abc"
    try:
        _ = "%x" % 3.0
    except TypeError as error:
        assert str(error) == "%x format: an integer is required, not float"
    else:
        raise AssertionError("%x accepted a float")


def test_dict_constructor_accepts_mappings_pairs_and_keywords():
    class Mapping:
        def keys(self):
            return ["x", "y"]

        def __getitem__(self, key):
            return key.upper()

    assert dict() == {}
    assert dict([(1, 2), ("a", "b")]) == {1: 2, "a": "b"}
    assert dict(zip("ab", [1, 2])) == {"a": 1, "b": 2}
    assert dict((key, key * 2) for key in range(3)) == {0: 0, 1: 2, 2: 4}
    assert dict(["ab", "cd"]) == {"a": "b", "c": "d"}
    assert dict({"a": 1}, b=2, a=3) == {"a": 3, "b": 2}
    assert dict(Mapping(), x=5) == {"x": 5, "y": "Y"}
    assert dict({1: 2}.items()) == {1: 2}
    assert list(dict([(2, "a"), (1, "b"), (2, "c")]).items()) == [(2, "c"), (1, "b")]
    source = {"a": 1}
    copy = dict(source)
    copy["b"] = 2
    assert source == {"a": 1}
    _assert_raises(lambda: dict([], []), TypeError, "dict expected at most 1 argument, got 2")
    _assert_raises(lambda: dict(5), TypeError, "'int' object is not iterable")
    _assert_raises(lambda: dict([(1, 2), 3]), TypeError, "object is not iterable")
    _assert_raises(
        lambda: dict([(1, 2, 3)]),
        ValueError,
        "dictionary update sequence element #0 has length 3; 2 is required",
    )
    _assert_raises(
        lambda: dict(["ab", "c"]),
        ValueError,
        "dictionary update sequence element #1 has length 1; 2 is required",
    )


def test_bytes_affixes_membership_find_and_join():
    for value in (b"abcab", bytearray(b"abcab")):
        assert value.startswith((b"x", b"ab")) and not value.startswith(())
        assert value.endswith(b"ab", 0, 5) and not value.endswith(b"ab", 0, 4)
        assert value.startswith(b"ca", 2) and value.endswith(bytearray(b"bc"), None, -2)
        assert b"bca" in value and b"" in value and b"cc" not in value
        assert bytearray(b"ca") in value and 99 in value and 100 not in value
        assert value.find(b"ab", 1) == 3 and value.find(b"ab", 1, 4) == -1
        assert value.find(b"", 5) == 5 and value.find(b"", 9) == -1
        joined = value.join([b"x", bytearray(b"y")])
        assert joined == b"xabcaby" and type(joined) is type(value)
    assert b", ".join(iter([b"a", b"b"])) == b"a, b"
    assert b"-".join([]) == b""
    _assert_raises(
        lambda: b"".join([b"a", "b"]),
        TypeError,
        "sequence item 1: expected a bytes-like object, str found",
    )
    _assert_raises(
        lambda: b"a".startswith("a"),
        TypeError,
        "startswith first arg must be bytes or a tuple of bytes, not str",
    )
    _assert_raises(lambda: b"a".endswith(("a",)), TypeError, "a bytes-like object is required, not 'str'")
    _assert_raises(lambda: "a" in b"a", TypeError, "a bytes-like object is required, not 'str'")
    _assert_raises(lambda: 256 in bytearray(b"a"), ValueError, "byte must be in range(0, 256)")
