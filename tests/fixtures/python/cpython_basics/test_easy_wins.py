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

    assert getattr(Item, "value") == 3
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
        assert False
    except ValueError:
        pass

    try:
        "".join([1])
        assert False
    except TypeError:
        pass

    try:
        [1].index(2)
        assert False
    except ValueError:
        pass

    try:
        {}.pop("missing")
        assert False
    except KeyError:
        pass
