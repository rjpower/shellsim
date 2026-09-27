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
