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


def test_comprehensions_evaluate_the_outermost_iterable_on_creation():
    values = iter([(0,), (1,)])
    # The generator iterates the old binding, not itself.
    values = (value + (None,) for value in values)
    assert next(values) == (0, None)
    numbers = [1, 2]
    doubled = (number * 2 for number in numbers)
    numbers = [5]
    assert list(doubled) == [2, 4]
    try:
        (value for value in 5)
    except TypeError as error:
        assert str(error) == "'int' object is not iterable"
    else:
        raise AssertionError("a generator accepted a non-iterable")

    class Table:
        rows = [1, 2, 3]
        doubled = [row * 2 for row in rows]
        keyed = {row: row for row in rows}
        listed = list(row for row in rows)  # noqa: C400

    assert Table.doubled == [2, 4, 6]
    assert Table.keyed == {1: 1, 2: 2, 3: 3}
    assert Table.listed == [1, 2, 3]


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


def test_displays_expand_starred_iterables_in_order():
    def letters():
        yield "a"
        yield "b"

    values = [1, 2]
    assert [*values, 3, *range(2)] == [1, 2, 3, 0, 1]
    assert (0, *letters()) == (0, "a", "b")
    assert {*values, 2, *"ab"} == {1, 2, "a", "b"}
    pair = *values, 5
    assert pair == (1, 2, 5)
    number = 5
    try:
        expanded = [*number]
    except TypeError:
        pass
    else:
        raise AssertionError(f"expanded a non-iterable into {expanded}")


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


def test_ellipsis_literal_is_the_builtin_singleton():
    value = ...
    assert value is Ellipsis
    assert repr(...) == "Ellipsis"
    assert str(Ellipsis) == "Ellipsis"
    assert f"{...}" == "Ellipsis"
    assert type(...).__name__ == "ellipsis"
    assert type(...)() is ...
    assert isinstance(Ellipsis, type(...))
    assert bool(...)
    assert ... == Ellipsis
    assert ... != 0
    assert ... != "..."
    assert ... is not None


def test_ellipsis_is_hashable_and_survives_containers():
    assert [..., 1] == [Ellipsis, 1]
    assert (1, ...)[1] is Ellipsis
    assert {...: "key", None: "none"}[Ellipsis] == "key"
    members = {...}
    members.add(Ellipsis)
    assert members == {Ellipsis}
    assert {(..., 1): "pair"}[(Ellipsis, 1)] == "pair"
    assert ... in (1, Ellipsis)
    assert list({...: 1}) == [Ellipsis]


def test_ellipsis_serves_as_a_body_and_an_argument():
    def placeholder(): ...

    def identity(value):
        return value

    assert placeholder() is None
    assert identity(...) is Ellipsis


def test_user_subscripts_receive_ellipsis_inside_tuple_keys():
    class Capture:
        def __getitem__(self, key):
            return key

    capture = Capture()
    assert capture[...] is Ellipsis
    assert capture[..., 0] == (Ellipsis, 0)
    assert capture[1, ...] == (1, Ellipsis)
    first, second = capture[..., 1:2]
    assert first is Ellipsis and (second.start, second.stop) == (1, 2)
    items = [1, 2]
    try:
        items[...]
    except TypeError as error:
        assert str(error) == "list indices must be integers or slices, not ellipsis"
    else:
        raise AssertionError("a list accepted an Ellipsis index")


def test_except_then_finally_runs_cleanup_once():
    events = []

    try:
        raise ValueError("handled")
    except ValueError:
        events.append("except")
    finally:
        events.append("finally")

    assert events == ["except", "finally"]


def test_generators_keep_active_exceptions_isolated_while_suspended():
    def suspended():
        try:
            raise ValueError("preserved")
        except ValueError:
            yield "paused"
            raise

    generator = suspended()
    assert next(generator) == "paused"

    try:
        raise RuntimeError("unrelated")
    except RuntimeError:
        pass

    try:
        next(generator)
    except ValueError as error:
        assert str(error) == "preserved"
    else:
        raise AssertionError("bare raise lost the generator's active exception")


def test_augmented_assignment_updates_mutable_operands_in_place():
    items = [1]
    alias = items
    items += "ab"
    items *= 2
    assert alias == [1, "a", "b", 1, "a", "b"]
    members = {1, 2}
    same = members
    members |= {3}
    members -= {1}
    assert same == {2, 3}
    mapping = {"a": 1}
    view = mapping
    mapping |= [("b", 2)]
    assert view == {"a": 1, "b": 2}
    pair = (1,)
    original = pair
    pair += (2,)
    assert original == (1,) and pair == (1, 2)


def test_augmented_assignment_prefers_in_place_methods():
    class Accumulator:
        def __init__(self):
            self.calls = []

        def __iadd__(self, other):
            self.calls.append(("iadd", other))
            return self

        def __matmul__(self, other):
            return ("matmul", other)

    accumulator = Accumulator()
    alias = accumulator
    accumulator += 5
    assert accumulator is alias and alias.calls == [("iadd", 5)]
    accumulator @= 2
    assert accumulator == ("matmul", 2)
    nothing = None
    try:
        nothing **= 2
    except TypeError as error:
        assert str(error) == "unsupported operand type(s) for **=: 'NoneType' and 'int'"
    else:
        raise AssertionError("None **= 2 succeeded")


def test_not_implemented_declines_binary_and_comparison_operators():
    assert repr(NotImplemented) == "NotImplemented"
    assert type(NotImplemented).__name__ == "NotImplementedType"
    assert type(NotImplemented)() is NotImplemented

    class Declines:
        def __eq__(self, other):
            return NotImplemented

        def __lt__(self, other):
            return NotImplemented

        def __add__(self, other):
            return NotImplemented

        def __iadd__(self, other):
            return NotImplemented

    class Reflects:
        def __radd__(self, other):
            return "radd"

        def __gt__(self, other):
            return "gt"

    declines = Declines()
    assert (declines == 1) is False
    assert (declines != 1) is True
    assert (declines == declines) is True
    assert declines + Reflects() == "radd"
    assert (declines < Reflects()) == "gt"
    try:
        declines < 1  # noqa: B015
    except TypeError as error:
        assert str(error) == "'<' not supported between instances of 'Declines' and 'int'"
    else:
        raise AssertionError("a declined ordering succeeded")
    try:
        declines += 1
    except TypeError as error:
        assert str(error) == "unsupported operand type(s) for +=: 'Declines' and 'int'"
    else:
        raise AssertionError("a declined in-place addition succeeded")
    try:
        bool(NotImplemented)
    except TypeError as error:
        assert str(error) == "NotImplemented should not be used in a boolean context"
    else:
        raise AssertionError("NotImplemented was used as a truth value")


def positional_only(first, second=2, /, third=3, *, fourth=4):
    return first, second, third, fourth


def positional_only_with_keywords(name, /, **options):
    return name, options


def test_positional_only_parameters_reject_keywords():
    assert positional_only(1) == (1, 2, 3, 4)
    assert positional_only(1, 5, third=6, fourth=7) == (1, 5, 6, 7)
    assert positional_only_with_keywords("x", name="y") == ("x", {"name": "y"})
    assert (lambda value, /: value * 2)(4) == 8
    try:
        positional_only(first=1)
    except TypeError as error:
        assert str(error) == (
            "positional_only() got some positional-only arguments passed as keyword arguments: 'first'"
        )
    else:
        raise AssertionError("a positional-only parameter was bound by keyword")


def test_container_repr_uses_item_repr_and_marks_self_references():
    class Item:
        def __repr__(self):
            return "Item()"

        def __str__(self):
            return "item"

    values = [Item(), (Item(),), {Item(): [Item()]}, frozenset(), set()]
    values.append(values)
    assert repr(values) == "[Item(), (Item(),), {Item(): [Item()]}, frozenset(), set(), [...]]"
    assert str([Item()]) == "[Item()]"
    assert f"{(Item(), 1)}" == "(Item(), 1)"
    mapping = {}
    mapping["self"] = mapping
    assert repr(mapping) == "{'self': {...}}"


def test_hash_follows_cpython_numeric_and_container_rules():
    # Numeric and numeric-tuple hashes are fixed by CPython's specification, independent of
    # PYTHONHASHSEED; string hashes are only required to be consistent.
    assert hash(-1) == -2
    assert hash(2**61) == 1
    assert hash(2**100) == 549755813888
    assert hash(2.0) == hash(2) == hash(2 + 0j)
    assert hash(1.5) == 1152921504606846977
    assert hash(float("inf")) == 314159
    assert hash(1 + 2j) == 2000007
    assert hash(None) == 4238894112
    assert hash((1, 2)) == -3550055125485641917
    assert hash(()) == 5740354900026072187
    assert hash(frozenset({1, 2, 3})) == -272375401224217160
    assert hash("text") == hash("te" + "xt")
    assert hash(b"abc") == hash(b"abc")
    assert hash(range(0, 10, 2)) == hash(range(0, 9, 2))

    class Eq:
        def __eq__(self, other):
            return True

    class Wide:
        def __hash__(self):
            return 2**64

    class MinusOne:
        def __hash__(self):
            return -1

    class Unhashable:
        __hash__ = None

    class Plain:
        pass

    assert hash(Wide()) == 8
    assert hash(MinusOne()) == -2
    plain = Plain()
    assert hash(plain) == hash(plain)
    for value, name in [
        ([], "list"),
        ({}, "dict"),
        (set(), "set"),
        (Eq(), "Eq"),
        (Unhashable(), "Unhashable"),
        ((1, [2]), "list"),
    ]:
        try:
            hash(value)
        except TypeError as error:
            assert str(error) == f"unhashable type: '{name}'"
        else:
            raise AssertionError(f"hash({value!r}) did not raise")


def test_float_repr_is_shortest_round_trip_with_ties_to_even():
    assert repr(0.1) == "0.1"
    assert repr(1e16) == "1e+16"
    assert repr(1e15) == "1000000000000000.0"
    assert repr(0.0001) == "0.0001"
    assert repr(0.00001) == "1e-05"
    assert repr(5e-324) == "5e-324"
    assert repr(-0.0) == "-0.0"
    # 16.5042266845703125 lies halfway between two 17-digit strings; the even one wins.
    assert repr(16.5042266845703125) == "16.504226684570312"
    assert str(-28.182485580444336) == "-28.182485580444336"
    assert repr(complex(1e16, 16.5042266845703125)) == "(1e+16+16.504226684570312j)"


def test_sum_compensates_float_and_complex_totals():
    assert sum([0.1] * 10) == 1.0
    assert sum([1e100, 1.0, -1e100]) == 1.0
    assert sum([0.1 + 0.2j] * 10) == 1 + 2j
    assert sum(x / 7 for x in range(1000)) == 71357.14285714286
    assert repr(sum([-0.0], -0.0)) == "-0.0"
    assert sum([float("inf"), 1.0]) == float("inf")
    assert sum([1, 2.5, 3]) == 6.5
    assert sum([2**63 - 1, 1]) == 2**63
    assert sum([], True) is True
    assert sum([[1], [2]], []) == [1, 2]
    for start, name in [
        ("", "strings [use ''.join(seq) instead]"),
        (b"", "bytes [use b''.join(seq) instead]"),
        (bytearray(), "bytearray [use b''.join(seq) instead]"),
    ]:
        try:
            sum([], start)
        except TypeError as error:
            assert str(error) == f"sum() can't sum {name}"
        else:
            raise AssertionError(f"sum() accepted start {start!r}")
    try:
        sum([1.0, 10**400])
    except OverflowError as error:
        assert str(error) == "int too large to convert to float"
    else:
        raise AssertionError("sum() did not overflow")


def test_with_enter_failure_reaches_the_enclosing_context_only():
    events = []

    class Outer:
        def __enter__(self):
            return self

        def __exit__(self, kind, value, traceback):
            events.append(("outer", kind.__name__, str(value)))
            return True

    class Inner:
        def __enter__(self):
            raise ValueError("boom")

        def __exit__(self, *args):
            events.append(("inner",))

    with Outer():
        with Inner():
            events.append(("body",))
    assert events == [("outer", "ValueError", "boom")]

    class NoEnter:
        def __exit__(self, *args):
            pass

    class NoExit:
        def __enter__(self):
            return self

    # CPython names user classes by module and qualified name, which shellsim does not model, so
    # only the builtin type's message is compared in full.
    for context, method in [(NoEnter(), "__enter__"), (NoExit(), "__exit__"), (1, "__exit__")]:
        try:
            with context:
                pass
        except TypeError as error:
            assert str(error).endswith(f"object does not support the context manager protocol (missed {method} method)")
        else:
            raise AssertionError(f"{context!r} was accepted as a context manager")
    try:
        with 1:
            pass
    except TypeError as error:
        assert str(error) == ("'int' object does not support the context manager protocol (missed __exit__ method)")


def test_object_instances_are_identity_sentinels():
    sentinel = object()
    other = object()
    assert type(sentinel) is object
    assert sentinel is sentinel and sentinel != other
    assert {sentinel: 1}[sentinel] == 1
    assert repr(sentinel).startswith("<object object at 0x")
    try:
        sentinel.name = 1
    except AttributeError as error:
        assert str(error) == ("'object' object has no attribute 'name' and no __dict__ for setting new attributes")
    else:
        raise AssertionError("object() accepted an attribute")
    try:
        _ = sentinel.name
    except AttributeError as error:
        assert str(error) == "'object' object has no attribute 'name'"
    else:
        raise AssertionError("object() has an attribute")


def test_divmod_uses_dunder_methods_and_cpython_messages():
    class Pair:
        def __divmod__(self, other):
            return ("divmod", other)

        def __rdivmod__(self, other):
            return ("rdivmod", other)

    class FloorOnly:
        def __floordiv__(self, other):
            return 1

        def __mod__(self, other):
            return 2

    assert divmod(Pair(), 3) == ("divmod", 3)
    assert divmod(3, Pair()) == ("rdivmod", 3)
    assert divmod(7, -2) == (-4, -1)
    assert divmod(-7.5, 2) == (-4.0, 0.5)
    try:
        divmod(FloorOnly(), 1)
    except TypeError as error:
        assert str(error) == "unsupported operand type(s) for divmod(): 'FloorOnly' and 'int'"
    else:
        raise AssertionError("divmod() fell back to // and %")
    for operation in [lambda: 1.0 // 0.0, lambda: 1.0 % 0.0, lambda: divmod(1.0, 0), lambda: 1 // 0.0]:
        try:
            operation()
        except ZeroDivisionError as error:
            assert str(error) == "division by zero"
        else:
            raise AssertionError("division by zero succeeded")
    try:
        0.0**-1
    except ZeroDivisionError as error:
        assert str(error) == "zero to a negative power"
    else:
        raise AssertionError("0.0 ** -1 succeeded")


def test_attribute_stores_on_builtin_values_raise_attribute_error():
    def store(target):
        target.upper = 1

    try:
        store("text")
    except AttributeError as error:
        assert str(error) == "'str' object attribute 'upper' is read-only"
    else:
        raise AssertionError("str accepted an attribute")
    try:
        (1.5).real = 2
    except AttributeError as error:
        assert str(error) == "attribute 'real' of 'float' objects is not writable"
    else:
        raise AssertionError("float.real was writable")
    try:
        (1).name = 2
    except AttributeError as error:
        assert str(error) == ("'int' object has no attribute 'name' and no __dict__ for setting new attributes")
    else:
        raise AssertionError("int accepted an attribute")


def test_raise_from_raises_the_new_exception_and_validates_the_cause():
    try:
        try:
            int("x")
        except ValueError as error:
            raise RuntimeError("wrapped") from error
    except RuntimeError as error:
        assert str(error) == "wrapped"
    else:
        raise AssertionError("raise from did not raise")
    try:
        raise ValueError("plain") from None
    except ValueError as error:
        assert str(error) == "plain"
    try:
        raise ValueError("bad cause") from 1
    except TypeError as error:
        assert str(error) == "exception causes must derive from BaseException"
    else:
        raise AssertionError("a non-exception cause was accepted")


def test_slice_builtin_builds_slices():
    assert slice(3) == slice(None, 3, None)
    part = slice(1, 5, 2)
    assert (part.start, part.stop, part.step) == (1, 5, 2)
    assert isinstance(part, slice) and type(part) is slice
    assert repr(part) == "slice(1, 5, 2)"
    assert [0, 1, 2, 3, 4, 5][part] == [1, 3]
    assert "abcdef"[slice(None, None, -2)] == "fdb"
    try:
        slice()
    except TypeError as error:
        assert str(error) == "slice expected at least 1 argument, got 0"
    else:
        raise AssertionError("slice() accepted no arguments")
