"""Portable language and builtin behavior exercised through Python assertions."""


def test_comprehensions_cover_clauses_filters_and_scopes():
    numbers = [1, 2, 3, 4]
    assert [number * 2 for number in numbers if number % 2 == 0] == [4, 8]
    assert [(left, right) for left in [1, 2] for right in [3, 4] if left < right] == [
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    ]
    assert {number * number for number in numbers if number > 2} == {9, 16}
    assert {str(number): number * number for number in numbers if number != 2} == {
        "1": 1,
        "3": 9,
        "4": 16,
    }

    item = 99

    def shifted(offset):
        return [item + offset for item in range(3)]

    assert shifted(10) == [10, 11, 12]
    assert item == 99


def test_generator_expressions_are_lazy_scoped_iterables():
    item = 50
    visited = []

    def observe():
        for value in range(3):
            visited.append(value)
            yield value

    expression = (value * 2 for value in observe())
    assert visited == []
    assert next(expression) == 0
    assert visited == [0]
    assert list(expression) == [2, 4]
    assert sum(value * value for value in range(5) if value % 2) == 10
    assert list(value + 1 for value in [1, 2, 3]) == [2, 3, 4]  # noqa: C400
    assert item == 50


def test_dataclasses_and_enums_use_normal_runtime_values():
    import dataclasses
    import enum

    @dataclasses.dataclass
    class Point:
        x: int
        y: int = 4

    class Color(enum.Enum):
        RED = 1
        GREEN = 2

    point = Point(2)
    assert point.x == 2
    assert point.y == 4
    assert Point(y=9, x=3).y == 9
    assert Color.RED.name == "RED"
    assert Color.RED.value == 1
    assert Color.RED is Color.RED
    assert [member.name for member in Color] == ["RED", "GREEN"]


def test_modern_typing_names_and_runtime_helpers_are_inert():
    from typing import (
        Annotated,
        Generator,
        Iterable,
        Literal,
        Mapping,
        Protocol,
        Self,
        Sequence,
        TypeVar,
        Union,
        cast,
        get_args,
        get_origin,
        overload,
        runtime_checkable,
    )

    marker = TypeVar("marker")
    assert str(marker) == "~marker"
    assert cast(int, "value") == "value"
    assert get_origin(marker) is None
    assert get_args(marker) == ()
    assert all(
        value is not None
        for value in [
            Annotated,
            Generator,
            Iterable,
            Literal,
            Mapping,
            Protocol,
            Self,
            Sequence,
            Union,
            overload,
            runtime_checkable,
        ]
    )


def test_exception_handlers_else_finally_and_with():
    events = []
    try:
        raise ValueError("bad")
    except (TypeError, ValueError) as error:
        events.append(str(error))
    else:
        events.append("wrong")

    def fail():
        try:
            raise RuntimeError("boom")
        finally:
            events.append("cleanup")

    try:
        fail()
    except Exception:
        events.append("caught")

    class Context:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, kind, value, traceback):
            events.append(kind is ValueError)
            return True

    with Context():
        raise ValueError("ignored")

    assert events[:4] == ["bad", "cleanup", "caught", "enter"]
    assert len(events) == 5


def test_function_defaults_and_argument_kinds():
    seed = 1

    def combine(left, right=seed, total=seed + 2):
        return left + right + total

    seed = 100
    assert combine(3) == 7
    assert combine(3, 4) == 10
    assert combine(left=3, total=9, right=8) == 20

    def append(value, bucket=[]):  # noqa: B006 - exercise Python's default binding semantics
        bucket.append(value)
        return bucket

    assert append(1) == [1]
    assert append(2) == [1, 2]

    def total(prefix, *values):
        return prefix + sum(values)

    def configure(prefix, *, window, scale=2):
        return prefix + window * scale

    def collect(prefix, *values, suffix):
        return prefix + sum(values) + suffix

    assert total(10) == 10
    assert total(1, 2, 3, 4) == 10
    assert configure(1, window=3) == 7
    assert collect(1, 2, 3, suffix=4) == 10
    assert (lambda *, value=5: value)(value=7) == 7


def test_sequence_operations_create_new_values():
    left = [1, 2]
    right = left + [3]
    repeated = ("x",) * 3
    right.append(4)
    assert left == [1, 2]
    assert right == [1, 2, 3, 4]
    assert repeated == ("x", "x", "x")


def test_generators_suspend_with_persistent_lexical_state():
    def values(limit):
        current = 0
        while current < limit:
            yield current * 2
            current += 1

    items = values(3)
    assert next(items) == 0
    assert next(items) == 2
    assert list(items) == [4]
    assert next(items, "done") == "done"

    def make(step):
        value = 1

        def sequence():
            nonlocal value
            yield value
            value += step
            yield value

        return sequence

    closure = make(4)()
    assert [next(closure), next(closure), next(closure, None)] == [1, 5, None]

    def delegated():
        yield from [1, 2, 3]

    assert list(delegated()) == [1, 2, 3]

    def receiver():
        received = yield "ready"
        yield received

    messages = receiver()
    assert messages.__next__() == "ready"
    assert messages.send("sent") == "sent"
    assert next(messages, "done") == "done"

    cleaned = []

    def closable():
        try:
            yield "open"
        finally:
            cleaned.append("closed")

    open_generator = closable()
    assert next(open_generator) == "open"
    assert open_generator.close() is None
    assert cleaned == ["closed"]
    assert next(open_generator, "done") == "done"

    thrown_cleanup = []

    def throwable():
        try:
            yield "open"
        finally:
            thrown_cleanup.append("closed")

    thrown = throwable()
    assert next(thrown) == "open"
    try:
        thrown.throw(ValueError)
    except ValueError:
        pass
    else:
        raise AssertionError("generator.throw did not raise")
    assert thrown_cleanup == ["closed"]


def test_numeric_literals_and_arithmetic_match_python():
    values = [1.2, 0.5, 1.0, 1_000.50_0, 1_2e-1, 1_2e1]
    assert values == [1.2, 0.5, 1.0, 1000.5, 1.2, 120.0]
    assert 1.2 + 0.5 == 1.7
    assert 1.2 * 2 == 2.4
    assert 5.0 / 2 == 2.5
    assert 1e9999 == float("inf")
    assert 1e-9999 == 0.0

    large = 9223372036854775808
    assert 9223372036854775807 + 1 == large
    assert large * large == 85070591730234615865843651857942052864
    assert (-large // 3, -large % 3) == (-3074457345618258603, 1)
    assert 9007199254740993 != 9007199254740992.0
    assert 9007199254740993 > 9007199254740992.0


def test_bytes_and_bytearray_preserve_octets():
    value = b"A\x00\xff\n"
    assert type(value) is bytes
    assert list(value) == [65, 0, 255, 10]
    assert value[1:3] == b"\x00\xff"
    assert value.hex() == "4100ff0a"
    assert "café".encode().decode() == "café"
    assert bytes([0, 127, 255]) == b"\x00\x7f\xff"
    assert b"caf\xe9".decode("latin-1") == "café"

    mutable = bytearray(b"ab")
    mutable.append(255)
    mutable.extend([0, 1])
    mutable[0] = 90
    assert list(mutable) == [90, 98, 255, 0, 1]
    assert mutable[1:4] == bytearray(b"b\xff\x00")
    assert bytes(mutable) == b"Zb\xff\x00\x01"


def test_mutable_sequence_slices_replace_delete_and_validate_atomically():
    value = bytearray(b"abcdef")
    value[2:4] = b"Q"
    value[1:2] = b"WXYZ"
    del value[2:6]
    assert value == bytearray(b"aWef")

    value = bytearray(b"abcdef")
    value[::-2] = b"XYZ"
    assert value == bytearray(b"aZcYeX")
    del value[1::2]
    assert value == bytearray(b"ace")

    value = bytearray(b"abcd")
    value[1:3] = value
    assert value == bytearray(b"aabcdd")
    for replacement in (b"xy", [1, 300]):
        before = bytes(value)
        try:
            value[::2] = replacement
            raise AssertionError("extended slice assignment accepted an invalid replacement")
        except ValueError:
            assert bytes(value) == before

    values = [0, 1, 2, 3, 4]
    values[1:4] = [8, 9]
    values[::2] = [5, 6]
    before = values.copy()
    try:
        values[::2] = [1]
        raise AssertionError("extended slice assignment accepted the wrong length")
    except ValueError:
        assert values == before
    del values[1::2]
    assert values == [5, 6]


def test_user_subscript_methods_receive_ordinary_slice_values():
    events = []

    class Capture:
        def __getitem__(self, key):
            events.append(("get", key.start, key.stop, key.step))
            return 7

        def __setitem__(self, key, value):
            events.append(("set", key.start, key.stop, key.step, value))

        def __delitem__(self, key):
            events.append(("del", key.start, key.stop, key.step))

    capture = Capture()
    assert capture[1:5:2] == 7
    capture[:3] = 9
    del capture[::-1]
    assert events == [
        ("get", 1, 5, 2),
        ("set", None, 3, None, 9),
        ("del", None, None, -1),
    ]
