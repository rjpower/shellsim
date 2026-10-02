# Portable CPython-style probes for common builtin and container behavior.


def raised(operation):
    try:
        operation()
    except Exception as error:
        return type(error), str(error)
    raise AssertionError("operation did not raise")


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


def test_dict_union_builds_a_new_dict_and_in_place_union_updates():
    left = {"a": 1, "b": 0}
    assert left | {"b": 2} == {"a": 1, "b": 2} and left == {"a": 1, "b": 0}
    alias = left
    left |= [("c", 3)]
    assert alias is left and left == {"a": 1, "b": 0, "c": 3}
    try:
        _ = {} | [1]
    except TypeError as error:
        assert str(error) == "unsupported operand type(s) for |: 'dict' and 'list'"
    else:
        raise AssertionError("dict | list did not raise TypeError")


def test_dict_pop_supports_defaults():
    value = {"a": 1}
    assert value.pop("a") == 1
    assert value.pop("missing", 3) == 3
    assert value == {}


def test_set_union_accepts_multiple_iterables():
    assert {1, 2}.union({2, 3}, [4]) == {1, 2, 3, 4}


def test_list_pop_and_insert_resolve_indices():
    values = [1, 2, 3, 4]
    assert values.pop(0) == 1
    assert values.pop(-2) == 3
    assert values.pop(True) == 4
    values.insert(-10, 0)
    values.insert(10, 9)
    assert values == [0, 2, 9]
    assert raised(lambda: values.pop(3)) == (IndexError, "pop index out of range")
    assert raised(lambda: values.pop(-4)) == (IndexError, "pop index out of range")
    assert raised(lambda: [].pop()) == (IndexError, "pop from empty list")
    assert raised(lambda: values.pop("0")) == (TypeError, "'str' object cannot be interpreted as an integer")
    assert raised(lambda: values.insert(1.0, 5)) == (TypeError, "'float' object cannot be interpreted as an integer")
    assert raised(lambda: values.remove(7)) == (ValueError, "list.remove(x): x not in list")
    assert values == [0, 2, 9]


def test_list_extend_and_sort_keep_cpython_order():
    values = ["bb", "a"]
    values.extend(word for word in ("cc", "d"))
    values.sort(key=len)
    assert values == ["a", "d", "bb", "cc"]
    values.sort(key=len, reverse=True)
    assert values == ["bb", "cc", "a", "d"]
    assert raised(lambda: values.extend(5)) == (TypeError, "'int' object is not iterable")


def test_dict_popitem_removes_the_newest_entry():
    value = {"a": 1, "b": 2, "c": 3}
    assert value.popitem() == ("c", 3)
    value["d"] = 4
    assert value.popitem() == ("d", 4)
    value["a"] = 5
    assert value.popitem() == ("b", 2)
    assert value.popitem() == ("a", 5)
    assert raised(value.popitem) == (KeyError, "'popitem(): dictionary is empty'")


def test_dict_clear_empties_every_alias():
    value = {"a": 1, (1, 2): [3]}
    alias = value
    assert value.clear() is None
    assert alias == {}
    alias["b"] = 2
    assert value == {"b": 2}


def test_dict_fromkeys_is_a_class_method():
    assert dict.fromkeys("abca") == {"a": None, "b": None, "c": None}
    assert list(dict.fromkeys([3, 1, 3, 2])) == [3, 1, 2]
    shared = []
    value = dict.fromkeys(("x", "y"), shared)
    assert value == {"x": [], "y": []}
    assert value["x"] is shared and value["y"] is shared
    # An instance lookup still binds the class, so the receiver's entries are not copied.
    assert {"ignored": 1}.fromkeys([1], 0) == {1: 0}
    assert dict.fromkeys({"k": 1}, 2) == {"k": 2}
    assert dict.fromkeys([]) == {}
    assert raised(lambda: dict.fromkeys(5)) == (TypeError, "'int' object is not iterable")


def test_dict_pop_raises_key_error_with_the_key():
    assert raised(lambda: {}.pop("missing")) == (KeyError, "'missing'")
    assert raised(lambda: {}.pop(3)) == (KeyError, "3")


def test_dict_views_follow_the_dict():
    value = {"a": 1, "b": 2}
    keys, values, items = value.keys(), value.values(), value.items()
    assert [type(view).__name__ for view in (keys, values, items)] == ["dict_keys", "dict_values", "dict_items"]
    value["c"] = 3
    del value["a"]
    assert (len(keys), list(keys), list(values), list(items)) == (2, ["b", "c"], [2, 3], [("b", 2), ("c", 3)])
    assert ("b" in keys, "a" in keys, 3 in values, 9 in values) == (True, False, True, False)
    assert (("c", 3) in items, ("c", 4) in items, "c" in items) == (True, False, False)
    assert list(reversed(keys)) == ["c", "b"]
    assert (bool({}.keys()), bool(keys)) == (False, True)
    assert (repr(items), str(keys)) == ("dict_items([('b', 2), ('c', 3)])", "dict_keys(['b', 'c'])")
    nested = {}
    nested["self"] = nested.values()
    assert repr(nested) == "{'self': dict_values([...])}"
    assert raised(lambda: keys[0]) == (TypeError, "'dict_keys' object is not subscriptable")
    assert raised(lambda: hash(keys)) == (TypeError, "unhashable type: 'dict_keys'")


def test_keys_and_items_views_are_set_like():
    keys = {"a": 1, "b": 2}.keys()
    assert keys == {"a", "b"} and {"a", "b"} == keys and keys != ["a", "b"]
    assert keys == {"b": 0, "a": 0}.keys()
    assert (keys & {"a", "z"}, {"a", "z"} & keys) == ({"a"}, {"a"})
    assert (keys | ["z"], keys ^ {"a", "z"}, keys - {"a"}, ["a", "z"] - keys) == (
        {"a", "b", "z"},
        {"b", "z"},
        {"b"},
        {"z"},
    )
    assert (keys <= {"a", "b"}, keys < {"a", "b"}, {"a"} < keys, keys >= {"a"}) == (True, False, True, True)
    assert (keys.isdisjoint(["x"]), keys.isdisjoint(["x", "a"])) == (True, False)
    assert raised(lambda: keys | 1) == (TypeError, "'int' object is not iterable")
    items = {"a": 1}.items()
    assert items == {("a", 1)} and items & {("a", 1), ("b", 2)} == {("a", 1)}
    values = {"a": 1}.values()
    assert values == values and values != {"a": 1}.values()


def test_objects_with_keys_and_getitem_are_mappings():
    class Mapping:
        def keys(self):
            return ["x", "y"]

        def __getitem__(self, key):
            return key.upper()

    def collect(**kwargs):
        return kwargs

    assert collect(**Mapping()) == {"x": "X", "y": "Y"}
    assert {**Mapping(), "z": 1} == {"x": "X", "y": "Y", "z": 1}
    updated = {}
    updated.update(Mapping())
    assert updated == {"x": "X", "y": "Y"}
    assert "%(x)s-%(y)s" % Mapping() == "X-Y"
    assert "{x}/{y}".format_map(Mapping()) == "X/Y"

    class Defaults(dict):
        def __missing__(self, key):
            return f"<{key}>"

    assert "{a} {b}".format_map(Defaults(a=1)) == "1 <b>"
    assert raised(lambda: {**[1]}) == (TypeError, "'list' object is not a mapping")
    assert raised(lambda: collect(**[1]))[0] is TypeError
    assert raised(lambda: collect(**{1: 2})) == (TypeError, "keywords must be strings")
    assert raised(lambda: collect(a=1, **{"a": 2}))[0] is TypeError
    assert raised(lambda: "%(x)s" % ("a",)) == (TypeError, "format requires a mapping")  # noqa: F502 - exercise the error


def test_set_pop_and_clear():
    values = {1, 2, 3}
    popped = values.pop()
    assert popped in {1, 2, 3}
    assert popped not in values
    assert len(values) == 2
    alias = values
    assert values.clear() is None
    assert alias == set()
    assert raised(values.pop) == (KeyError, "'pop from an empty set'")


def test_set_algebra_accepts_any_iterables():
    base = {1, 2, 3, 4}
    assert base.intersection([2, 3, 5], (3, 4, 2)) == {2, 3}
    assert base.intersection(range(3)) == {1, 2}
    assert base.difference([1], {4, 9}) == {2, 3}
    assert base.symmetric_difference([3, 4, 5, 5]) == {1, 2, 5}
    assert base.symmetric_difference(value for value in [1, 6]) == {2, 3, 4, 6}
    copy = base.intersection()
    assert copy == base
    assert copy is not base
    assert base.difference() == base
    assert base == {1, 2, 3, 4}


def test_set_update_methods_mutate_in_place():
    values = {1, 2, 3, 4}
    assert values.intersection_update([1, 2, 3], {2, 3, 4}) is None
    assert values == {2, 3}
    assert values.difference_update([3], (7,)) is None
    assert values == {2}
    assert values.symmetric_difference_update([2, 5, 5, 6]) is None
    assert values == {5, 6}
    assert values.update([7], (8,)) is None
    assert values.update() is None
    assert values == {5, 6, 7, 8}
    values.intersection_update(values)
    assert values == {5, 6, 7, 8}
    values.symmetric_difference_update(values)
    assert values == set()
    values.update("ab")
    values.difference_update(values)
    assert values == set()


def test_set_comparisons_accept_any_iterables():
    assert {1, 2}.issubset([1, 2, 3])
    assert not {1, 4}.issubset(range(3))
    assert set().issubset([])
    assert {1, 2, 3}.issuperset([1, 3, 3])
    assert not {1}.issuperset([1, 2])
    assert {1, 2}.isdisjoint([3, 4])
    assert not {1, 2}.isdisjoint(value for value in [5, 2])
    assert set().isdisjoint(set())


def test_set_comparisons_stop_consuming_an_iterable_early():
    items = iter([1, 5, 2, 3])
    assert not {1, 2}.issuperset(items)
    assert list(items) == [2, 3]
    items = iter([5, 2, 3])
    assert not {1, 2}.isdisjoint(items)
    assert list(items) == [3]


def test_frozenset_methods_return_frozensets():
    base = frozenset([1, 2, 3])
    assert base.intersection([2, 3, 4]) == frozenset([2, 3])
    assert base.difference([1]) == frozenset([2, 3])
    assert base.symmetric_difference([3, 4]) == frozenset([1, 2, 4])
    for result in (
        base.intersection([2]),
        base.difference([1]),
        base.symmetric_difference([3]),
        base.union([9]),
        base.copy(),
    ):
        assert type(result) is frozenset
    assert type({1}.intersection(frozenset([1]))) is set
    assert base.issubset([1, 2, 3, 4])
    assert base.issuperset({1})
    assert base.isdisjoint([7])
    assert base == frozenset([1, 2, 3])


def test_set_methods_reject_bad_operands():
    for operation in (
        lambda: set().intersection([1], 5),
        lambda: set().difference(5),
        lambda: {1}.symmetric_difference(5),
        lambda: {1}.intersection_update(5),
        lambda: {1}.difference_update(5),
        lambda: {1}.symmetric_difference_update(5),
        lambda: {1}.update([2], 5),
        lambda: {1}.issubset(5),
        lambda: {1}.issuperset(5),
        lambda: {1}.isdisjoint(5),
        lambda: frozenset().intersection(5),
    ):
        assert raised(operation) == (TypeError, "'int' object is not iterable")
    assert raised(lambda: set.add(frozenset(), 1)) == (
        TypeError,
        "descriptor 'add' for 'set' objects doesn't apply to a 'frozenset' object",
    )
    assert raised(lambda: set.clear(frozenset([1]))) == (
        TypeError,
        "descriptor 'clear' for 'set' objects doesn't apply to a 'frozenset' object",
    )
    assert raised(lambda: {1}.remove(2)) == (KeyError, "2")


def test_bytearray_mutators_edit_in_place():
    data = bytearray(b"abc")
    alias = data
    data.insert(0, ord("z"))
    data.insert(-1, 0x2D)
    data.insert(99, 0x21)
    assert data == bytearray(b"zab-c!")
    assert data.pop() == ord("!")
    assert data.pop(0) == ord("z")
    assert data.pop(-2) == 0x2D
    data.remove(ord("b"))
    assert alias == bytearray(b"ac")
    copy = data.copy()
    data.clear()
    assert alias == bytearray()
    assert copy == bytearray(b"ac")
    assert type(copy) is bytearray


def test_bytearray_mutators_raise_cpython_errors():
    data = bytearray(b"a")
    assert raised(lambda: bytearray().pop()) == (IndexError, "pop from empty bytearray")
    assert raised(lambda: data.pop(3)) == (IndexError, "pop index out of range")
    assert raised(lambda: data.remove(ord("z"))) == (ValueError, "value not found in bytearray")
    assert raised(lambda: data.remove(256)) == (ValueError, "byte must be in range(0, 256)")
    assert raised(lambda: data.insert(0, -1)) == (ValueError, "byte must be in range(0, 256)")
    assert raised(lambda: data.append("b")) == (TypeError, "'str' object cannot be interpreted as an integer")
    assert raised(lambda: data.extend(["b"])) == (TypeError, "'str' object cannot be interpreted as an integer")
    assert data == bytearray(b"a")


def test_bytes_like_search_and_affix_methods():
    for data in (b"banana", bytearray(b"banana")):
        assert data.startswith(b"ban")
        assert data.endswith(bytearray(b"na"))
        assert data.index(b"an") == 1
        assert data.index(b"an", 2) == 3
        assert data.index(b"") == 0
        assert raised(lambda data=data: data.index(b"x")) == (ValueError, "subsection not found")


def test_bytes_like_strip_split_and_join():
    for kind in (bytes, bytearray):
        data = kind(b" \t\x0b\x0ca b\r\n ")
        assert data.strip() == b"a b"
        assert data.lstrip() == b"a b\r\n "
        assert data.rstrip() == b" \t\x0b\x0ca b"
        assert type(data.strip()) is kind
        assert kind(b"xxabyx").strip(b"xy") == b"ab"
        assert kind(b"xxab").lstrip(None) == b"xxab"
        assert kind(b" a  b ").split() == [b"a", b"b"]
        assert kind(b" a b c ").split(None, 1) == [b"a", b"b c "]
        assert kind(b"a,,b").split(b",") == [b"a", b"", b"b"]
        assert kind(b"a,b,c").split(b",", 1) == [b"a", b"b,c"]
        assert kind(b"").split() == []
        assert kind(b"").split(b",") == [b""]
        assert all(type(part) is kind for part in kind(b"a b").split())
        assert kind(b", ").join([b"a", bytearray(b"b"), kind(b"c")]) == b"a, b, c"
        assert type(kind(b",").join([])) is kind
    assert raised(lambda: b"a".split(b"")) == (ValueError, "empty separator")
    assert raised(lambda: b",".join([b"a", "b"])) == (
        TypeError,
        "sequence item 1: expected a bytes-like object, str found",
    )


def test_bytes_like_case_and_replace():
    for kind in (bytes, bytearray):
        assert kind(b"aB\xe9").upper() == b"AB\xe9"
        assert kind(b"aB\xe9").lower() == b"ab\xe9"
        assert type(kind(b"a").upper()) is kind
        assert kind(b"aaaa").replace(b"a", b"b", 2) == b"bbaa"
        assert kind(b"aaaa").replace(b"aa", b"c") == b"cc"
        assert kind(b"ab").replace(b"", b"-") == b"-a-b-"
        assert kind(b"ab").replace(b"", b"-", 2) == b"-a-b"
        assert kind(b"abc").replace(b"b", b"") == b"ac"
        assert kind(b"abc").replace(b"x", b"y", -1) == b"abc"
        assert type(kind(b"a").replace(b"a", b"b")) is kind


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


COMPLEX_FORMAT_CASES = [
    (1 + 2j, "", "(1+2j)"),
    (2j, "+", "+2j"),
    (complex(-0.0, 2), "", "(-0+2j)"),
    (complex(-0.0, -2), "z", "(0-2j)"),
    (1 + 2j, "#", "(1.+2.j)"),
    (1 + 2j, "10", "    (1+2j)"),
    (1 + 2j, ">14.1f", "      1.0+2.0j"),
    (1 + 2j, "*^12.1f", "**1.0+2.0j**"),
    (1234.5 - 5678.5j, ".2", "(1.2e+03-5.7e+03j)"),
    (1234.5 - 5678.5j, ".2g", "1.2e+03-5.7e+03j"),
    (1234.5 - 5678.5j, "+.2f", "+1234.50-5678.50j"),
    (1234.5 - 5678.5j, ",.1f", "1,234.5-5,678.5j"),
    (complex(-0.0, -0.0), ".1f", "-0.0-0.0j"),
    (1 + 2j, ".1e", "1.0e+00+2.0e+00j"),
    (complex(float("nan"), -float("inf")), "F", "NAN-INFj"),
    (complex(1, -float("nan")), "", "(1+nanj)"),
]

COMPLEX_FORMAT_ERRORS = [
    ("010.2f", "Zero padding is not allowed in complex format specifier"),
    ("0>10", "Zero padding is not allowed in complex format specifier"),
    ("=12.2f", "'=' alignment flag is not allowed in complex format specifier"),
    (".2%", "Unknown format code '%' for object of type 'complex'"),
    ("d", "Unknown format code 'd' for object of type 'complex'"),
    (",n", "Cannot specify ',' with 'n'."),
]


def test_complex_format_specifications_match_cpython():
    for value, spec, expected in COMPLEX_FORMAT_CASES:
        assert (value, spec, format(value, spec)) == (value, spec, expected)
    assert f"{3.14159 + 2.71828j:.2f}" == "3.14+2.72j"
    assert "{:.1f}".format(1 + 2j) == "1.0+2.0j"
    for spec, message in COMPLEX_FORMAT_ERRORS:
        try:
            format(1 + 2j, spec)
        except ValueError as error:
            assert (spec, str(error)) == (spec, message)
        else:
            raise AssertionError(spec)


def test_math_isclose_binds_arguments_as_cpython_does():
    import math

    assert math.isclose(1, 1 + 1e-10) and not math.isclose(0, 1e-10)
    assert math.isclose(0, 1e-10, abs_tol=1e-9) and math.isclose(a=1, b=1.0)
    assert math.isclose(math.inf, math.inf) and not math.isclose(math.nan, math.nan)
    assert not math.isclose(1, 2, rel_tol=math.nan)
    calls = [
        (lambda: math.isclose(1), TypeError, "isclose() missing required argument 'b' (pos 2)"),
        (lambda: math.isclose(1, 2, 3), TypeError, "isclose() takes exactly 2 positional arguments (3 given)"),
        (lambda: math.isclose(1, 2, tol=1), TypeError, "isclose() got an unexpected keyword argument 'tol'"),
        (lambda: math.isclose(1, 2, rel_tol=-1), ValueError, "tolerances must be non-negative"),
    ]
    for call, kind, message in calls:
        _assert_raises(call, kind, message)


def test_math_hyperbolic_and_gamma_functions():
    import math

    close = [
        (math.sinh(1), 1.1752011936438014),
        (math.cosh(1), 1.5430806348152437),
        (math.tanh(1), 0.7615941559557649),
        (math.asinh(1), 0.881373587019543),
        (math.acosh(2), 1.3169578969248166),
        (math.atanh(0.5), 0.5493061443340549),
        (math.gamma(0.5), math.sqrt(math.pi)),
        (math.gamma(-0.5), -2 * math.sqrt(math.pi)),
        (math.lgamma(-0.5), math.log(2 * math.sqrt(math.pi))),
        (math.lgamma(5), math.log(24)),
    ]
    for value, expected in close:
        assert math.isclose(value, expected, rel_tol=1e-15), (value, expected)
    assert (math.gamma(5), math.lgamma(1), math.lgamma(2)) == (24.0, 0.0, 0.0)
    inf = math.inf
    assert (math.sinh(-inf), math.cosh(-inf), math.tanh(-inf), math.asinh(-inf)) == (-inf, inf, -1.0, -inf)
    assert (math.acosh(inf), math.gamma(inf), math.lgamma(-inf)) == (inf, inf, inf)
    assert str(math.sinh(-0.0)) == "-0.0"
    for name in ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "gamma", "lgamma"]:
        assert math.isnan(getattr(math, name)(math.nan))
    failures = [
        (math.acosh, 0.5, ValueError),
        (math.atanh, 1.0, ValueError),
        (math.atanh, -inf, ValueError),
        (math.gamma, 0.0, ValueError),
        (math.gamma, -2.0, ValueError),
        (math.gamma, -inf, ValueError),
        (math.lgamma, -1.0, ValueError),
        (math.gamma, 172.0, OverflowError),
        (math.lgamma, 1e308, OverflowError),
        (math.sinh, 1000.0, OverflowError),
        (math.cosh, -1000.0, OverflowError),
    ]
    for function, argument, kind in failures:
        try:
            function(argument)
        except kind:
            pass
        else:
            raise AssertionError((function, argument))


class Rounded:
    def __round__(self, ndigits=None):
        return ("round", ndigits)

    def __floor__(self):
        return "floor"

    def __ceil__(self):
        return "ceil"

    def __trunc__(self):
        return "trunc"


class FloatLike:
    def __float__(self):
        return 2.5


def test_rounding_functions_defer_to_special_methods():
    import math

    value = Rounded()
    assert (round(value), round(value, 2), round(value, ndigits=-1)) == (("round", None), ("round", 2), ("round", -1))
    assert (math.floor(value), math.ceil(value), math.trunc(value)) == ("floor", "ceil", "trunc")
    assert (math.floor(FloatLike()), math.ceil(FloatLike()), math.sqrt(FloatLike())) == (2, 3, 2.5**0.5)
    calls = [
        (lambda: round(1j), "type complex doesn't define __round__ method"),
        (lambda: round(FloatLike()), "type FloatLike doesn't define __round__ method"),
        (lambda: math.trunc(1j), "type complex doesn't define __trunc__ method"),
        (lambda: math.trunc(FloatLike()), "type FloatLike doesn't define __trunc__ method"),
        (lambda: math.floor(1j), "must be real number, not complex"),
        (lambda: math.ceil(object()), "must be real number, not object"),
        (lambda: math.sqrt("4"), "must be real number, not str"),
    ]
    for call, message in calls:
        _assert_raises(call, TypeError, message)


INF = float("inf")
NAN = float("nan")

# (dividend, divisor, repr of divmod) recorded from CPython 3.14.
FLOAT_DIVMOD_CASES = [
    (-7.0, 2.0, "(-4.0, 1.0)"),
    (7.0, -2.0, "(-4.0, -1.0)"),
    (2.0, -0.5, "(-4.0, -0.0)"),
    (0.0, -1.0, "(-0.0, -0.0)"),
    (1.0, 0.1, "(9.0, 0.09999999999999995)"),
    (2.0, INF, "(0.0, 2.0)"),
    (-2.0, INF, "(-1.0, inf)"),
    (2.0, -INF, "(-1.0, -inf)"),
    (-0.0, INF, "(-0.0, 0.0)"),
    (INF, 0.5, "(nan, nan)"),
    (NAN, 2.0, "(nan, nan)"),
]


def test_float_floor_division_and_remainder_take_the_divisor_sign():
    for left, right, expected in FLOAT_DIVMOD_CASES:
        assert repr(divmod(left, right)) == expected, (left, right)
        assert repr((left // right, left % right)) == expected, (left, right)
    _assert_raises(lambda: 1.0 // 0.0, ZeroDivisionError, "division by zero")
    _assert_raises(lambda: 1.0 % -0.0, ZeroDivisionError, "division by zero")


def test_float_power_follows_ieee_special_values():
    cases = [
        (NAN**0.0, "1.0"),
        (1.0**NAN, "1.0"),
        (2.0**NAN, "nan"),
        (NAN**2.0, "nan"),
        ((-INF) ** 0.5, "inf"),
        ((-INF) ** -0.5, "0.0"),
        ((-INF) ** 3.0, "-inf"),
        (0.0**-INF, "inf"),
        (0.5**INF, "0.0"),
        ((-8.0) ** (1 / 3), "(1.0000000000000002+1.7320508075688772j)"),
    ]
    for value, expected in cases:
        assert repr(value) == expected, (value, expected)
    _assert_raises(lambda: 0.0**-1.0, ZeroDivisionError, "zero to a negative power")
    _assert_raises(lambda: 10.0**400.0, OverflowError, "(34, 'Numerical result out of range')")


def test_modular_pow_with_negative_exponent_inverts_the_base():
    assert pow(3, -1, 7) == 5
    assert pow(3, -2, 7) == 4
    assert pow(3, -1, -7) == -2
    assert pow(-3, -1, 7) == 2
    assert pow(2, -1, 1) == 0
    assert pow(10**30 + 1, -1, 10**20 + 7) == 18895348837412790699
    message = "base is not invertible for the given modulus"
    _assert_raises(lambda: pow(2, -1, 4), ValueError, message)
    _assert_raises(lambda: pow(0, -1, 5), ValueError, message)


class DeclinesOrdering:
    def __lt__(self, other):
        return NotImplemented


def test_ordering_complex_numbers_reports_the_comparison_as_written():
    calls = [
        (lambda: 1 < 1j, "'<' not supported between instances of 'int' and 'complex'"),
        (lambda: 1j >= 2j, "'>=' not supported between instances of 'complex' and 'complex'"),
        (lambda: DeclinesOrdering() < 1j, "'<' not supported between instances of 'DeclinesOrdering' and 'complex'"),
        (lambda: 1j > DeclinesOrdering(), "'>' not supported between instances of 'complex' and 'DeclinesOrdering'"),
    ]
    for call, message in calls:
        _assert_raises(call, TypeError, message)


def test_format_builtin_checks_its_arguments_as_cpython_does():
    assert (format(3.5), format(42, "#06x"), format("x", ">5s")) == ("3.5", "0x002a", "    x")
    calls = [
        (lambda: format(), "format expected at least 1 argument, got 0"),
        (lambda: format(1, "", ""), "format expected at most 2 arguments, got 3"),
        (lambda: format(1, 2), "format() argument 2 must be str, not int"),
        (lambda: format(1, spec="d"), "format() takes no keyword arguments"),
    ]
    for call, message in calls:
        try:
            call()
        except TypeError as error:
            assert str(error) == message
        else:
            raise AssertionError(message)


def test_int_and_float_find_conversion_methods_on_the_type():
    class Number:
        def __int__(self):
            return 10**100

        def __float__(self):
            return 0.5

    number = Number()
    number.__int__ = lambda: 7
    number.__float__ = lambda: 7.0
    assert (int(number), float(number)) == (10**100, 0.5)

    class Bad:
        def __int__(self):
            return "1"

        def __float__(self):
            return 1

    for conversion, message in [
        (int, "__int__ returned non-int (type str)"),
        (float, "Bad.__float__ returned non-float (type int)"),
    ]:
        try:
            conversion(Bad())
        except TypeError as error:
            assert str(error) == message
        else:
            raise AssertionError(message)


def test_dict_constructor_accepts_mappings_pairs_and_keywords():
    class Mapping:
        def keys(self):
            return ["x", "y"]

        def __getitem__(self, key):
            return key.upper()

    assert dict() == {}  # noqa: C408 - exercise the dict constructor
    assert dict([(1, 2), ("a", "b")]) == {1: 2, "a": "b"}  # noqa: C406 - exercise the dict constructor
    assert dict(zip("ab", [1, 2])) == {"a": 1, "b": 2}
    assert dict((key, key * 2) for key in range(3)) == {0: 0, 1: 2, 2: 4}  # noqa: C402 - exercise the dict constructor
    assert dict(["ab", "cd"]) == {"a": "b", "c": "d"}
    assert dict({"a": 1}, b=2, a=3) == {"a": 3, "b": 2}
    assert dict(Mapping(), x=5) == {"x": 5, "y": "Y"}
    assert dict({1: 2}.items()) == {1: 2}
    assert list(dict([(2, "a"), (1, "b"), (2, "c")]).items()) == [(2, "c"), (1, "b")]  # noqa: C406 - exercise the dict constructor
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


def test_int_to_bytes_and_from_bytes_match_cpython():
    assert (1024).to_bytes(2) == b"\x04\x00"
    assert (1024).to_bytes(2, byteorder="little") == b"\x00\x04"
    assert (-1).to_bytes(2, "big", signed=True) == b"\xff\xff"
    assert (0).to_bytes(0) == b"" and (5).to_bytes() == b"\x05" and True.to_bytes() == b"\x01"
    assert (2**70).to_bytes(10, "little") == b"\x00" * 8 + b"@\x00"
    assert int.from_bytes(b"\x00\x10") == 16
    assert int.from_bytes(b"\x00\x10", byteorder="little") == 4096
    assert int.from_bytes(b"\xff\xff", "big", signed=True) == -1
    assert int.from_bytes([1, 2]) == 258 and int.from_bytes(bytearray(b"\x01")) == 1
    assert int.from_bytes(b"") == 0 and int.from_bytes(b"", signed=True) == 0
    assert bool.from_bytes(b"\x01") is True and bool.from_bytes(b"\x00") is False
    for value in [0, 1, 127, -128, 255, -(2**63), 2**64 + 3]:
        encoded = value.to_bytes(9, "little", signed=True)
        assert int.from_bytes(encoded, "little", signed=True) == value


def test_int_byte_conversions_raise_cpython_errors():
    too_big = (OverflowError, "int too big to convert")
    assert raised(lambda: (-129).to_bytes(1, "big", signed=True)) == too_big
    assert raised(lambda: (128).to_bytes(1, "big", signed=True)) == too_big
    assert raised(lambda: (256).to_bytes(1)) == too_big
    assert raised(lambda: (-1).to_bytes(2)) == (OverflowError, "can't convert negative int to unsigned")
    assert raised(lambda: (1).to_bytes(-1)) == (ValueError, "length argument must be non-negative")
    order = (ValueError, "byteorder must be either 'little' or 'big'")
    assert raised(lambda: (1).to_bytes(1, "middle")) == order
    assert raised(lambda: int.from_bytes(b"\x01", "middle")) == order
    assert raised(lambda: (1).to_bytes(1, 5)) == (TypeError, "to_bytes() argument 'byteorder' must be str, not int")
    assert raised(lambda: (1).to_bytes(1.5)) == (TypeError, "'float' object cannot be interpreted as an integer")
    assert raised(lambda: (1).to_bytes(1, "big", True)) == (
        TypeError,
        "to_bytes() takes at most 2 positional arguments (3 given)",
    )
    assert raised(lambda: int.from_bytes()) == (TypeError, "from_bytes() missing required argument 'bytes' (pos 1)")
    assert raised(lambda: int.from_bytes(5)) == (TypeError, "cannot convert 'int' object to bytes")
    assert raised(lambda: int.from_bytes("ab")) == (TypeError, "cannot convert 'str' object to bytes")
    assert raised(lambda: int.from_bytes([256])) == (ValueError, "bytes must be in range(0, 256)")


def test_int_bit_and_ratio_methods():
    assert (0).bit_length() == 0 and (-255).bit_length() == 8 and (2**100).bit_length() == 101
    assert True.bit_length() == 1
    assert (-255).bit_count() == 8 and (2**100 - 1).bit_count() == 100
    assert (7).as_integer_ratio() == (7, 1) and True.as_integer_ratio() == (1, 1)
    assert (2**80).as_integer_ratio() == (2**80, 1)
    assert (7).is_integer() is True
    assert raised(lambda: (7).bit_length(1)) == (TypeError, "int.bit_length() takes no arguments (1 given)")


def test_int_subclasses_inherit_native_int_methods():
    class Small(int):
        pass

    class Custom(int):
        def bit_length(self):
            return "custom"

    assert Small(5).bit_length() == 3 and Small(12).to_bytes(2, "little") == b"\x0c\x00"
    assert type(Small(3).conjugate()) is int and type(Small(3).real) is int
    assert Small.bit_length(Small(5)) == 3
    restored = Small.from_bytes(b"\x03")
    assert type(restored) is Small and restored == 3
    assert Custom(3).bit_length() == "custom"
    assert not hasattr(Small(1), "missing")


def test_bytes_constructor_rejects_bad_items():
    assert raised(lambda: bytes([1, "a"])) == (TypeError, "'str' object cannot be interpreted as an integer")
    assert raised(lambda: bytes([256])) == (ValueError, "bytes must be in range(0, 256)")
    assert raised(lambda: bytes([2**70])) == (ValueError, "bytes must be in range(0, 256)")
    assert raised(lambda: bytes(-1)) == (ValueError, "negative count")


def test_center_puts_the_odd_padding_unit_where_cpython_does():
    assert "ab".center(5, "*") == "**ab*"
    assert "a".center(4, "*") == "*a**"
    assert "abc".center(6, "*") == "*abc**"
    assert "".center(3, "*") == "***"
    assert b"ab".center(5, b"*") == b"**ab*"
    assert bytearray(b"a").center(4, b"*") == bytearray(b"*a**")


def test_callable_recognizes_call_methods_and_native_methods():
    class Callable:
        def __call__(self):
            return 1

    class Inherits(Callable):
        pass

    assert callable(Callable()) and callable(Inherits())
    assert not callable(object())
    assert callable(str.upper) and callable("a".upper) and callable(dict.fromkeys)


def test_slice_indices_normalizes_bounds_for_a_length():
    assert slice(None).indices(5) == (0, 5, 1)
    assert slice(None, None, -1).indices(5) == (4, -1, -1)
    assert slice(-3, -1).indices(5) == (2, 4, 1)
    assert slice(-100, 100, 2).indices(5) == (0, 5, 2)
    assert slice(2, -100, -1).indices(5) == (2, -1, -1)
    assert slice(None, None, -1).indices(0) == (-1, -1, -1)
    assert raised(lambda: slice(None, None, 0).indices(3)) == (ValueError, "slice step cannot be zero")
    assert raised(lambda: slice(1).indices(-1)) == (ValueError, "length should not be negative")
    assert raised(lambda: slice(1).indices(1.5)) == (TypeError, "'float' object cannot be interpreted as an integer")


def test_dict_and_set_deletion_keep_order_and_lookup():
    mapping = {key: key * 2 for key in range(1000)}
    for key in range(0, 1000, 3):
        del mapping[key]
    assert len(mapping) == 666 and list(mapping)[:3] == [1, 2, 4]
    mapping[3] = "back"
    assert list(mapping)[-1] == 3 and 0 not in mapping and mapping[4] == 8
    assert mapping.popitem() == (3, "back") and mapping.pop(4) == 8

    members = set(range(100))
    for value in range(0, 100, 2):
        members.discard(value)
    members.remove(99)
    assert len(members) == 49 and 97 in members and 98 not in members
    while len(members) > 1:
        members.pop()
    assert len(members) == 1

    large = {key: key for key in range(100_000)}
    for key in range(99_999):
        del large[key]
    assert large == {99_999: 99_999}


def test_set_algebra_keeps_builtin_membership_and_kind():
    class AlwaysContains(set):
        def __contains__(self, value):
            return True

    assert {1, 2} & AlwaysContains({2, 3}) == {2}
    assert type(frozenset({1}) | {2}) is frozenset and type({2} | frozenset({1})) is set
    left, right = set(range(50_000)), set(range(25_000, 75_000))
    assert len(left | right) == 75_000 and len(left & right) == 25_000
    assert len(left - right) == 25_000 and len(left ^ right) == 50_000
    assert left.issubset(left | right) and len(left.union(right, [-1])) == 75_001


def test_int_decimal_conversion_is_limited_to_4300_digits():
    largest = 10**4300 - 1
    assert len(str(largest)) == 4300 and int("9" * 4300) == largest
    too_long = 10**4300
    for convert in (str, repr, "{}".format, "{:,}".format, "%d".__mod__, lambda value: str([value])):
        assert raised(lambda convert=convert: convert(too_long))[0] is ValueError
    assert raised(lambda: int("1" * 4301))[0] is ValueError
    assert hex(too_long).startswith("0x1") and int("f" * 5000, 16) > too_long
    assert (too_long * too_long) // too_long == too_long
