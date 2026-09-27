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
