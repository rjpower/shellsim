import json


class UserId(int):
    def next_id(self):
        return self + 1


class Base:
    label = "base"

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def describe(self):
        return "Base"


class Child(Base):
    label = "child"

    def describe(self):
        return super().describe() + " Child"


class ParentError(Exception):
    pass


class ChildError(ParentError):
    pass


class Reflected:
    def __radd__(self, left):
        return left + 10

    def __rsub__(self, left):
        return left - 10

    def __rmul__(self, left):
        return left * 10


metaclass_events = []


class Meta(type):
    def __prepare__(name, bases):
        metaclass_events.append("prepare")
        return {"prepared": True}

    def __new__(mcls, name, bases, namespace):
        metaclass_events.append("new")
        return super().__new__(mcls, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        metaclass_events.append("init")


class MetaProduct(metaclass=Meta):
    pass


def test_scalar_storage_has_one_semantic_type():
    short = "123456789012345"
    long = "1234567890123456"
    assert type(short) is str
    assert type(long) is str
    assert long + long == "12345678901234561234567890123456"
    assert UserId(12).next_id() == 13
    assert isinstance(UserId(1), int)


def test_binary_operations_dispatch_through_type_slots():
    value = Reflected()
    assert 2 + value == 12
    assert 20 - value == 10
    assert 3 * value == 30
    assert UserId(2) * "ab" == "abab"
    assert UserId(2) * [1] == [1, 1]
    assert (1,) * UserId(2) == (1, 1)


def test_descriptors_and_super_share_the_mro():
    value = Child(4)
    assert value.value == 4
    value.value = 9
    assert value.value == 9
    assert value.describe() == "Base Child"
    assert Child.describe(value) == "Base Child"


def test_exception_subclasses_inherit_argument_storage_and_rendering():
    empty = ParentError()
    single = ParentError("boom")
    multiple = ChildError("bad", 7)
    assert empty.args == ()
    assert str(empty) == ""
    assert repr(empty) == "ParentError()"
    assert single.args == ("boom",)
    assert str(single) == "boom"
    assert repr(single) == "ParentError('boom')"
    assert multiple.args == ("bad", 7)
    assert str(multiple) == "('bad', 7)"
    assert repr(multiple) == "ChildError('bad', 7)"

    try:
        raise multiple
    except ParentError as observed:
        assert observed is multiple


def test_metaclass_hooks_use_the_shared_allocator():
    assert metaclass_events == ["prepare", "new", "init"]
    assert MetaProduct.prepared is True
    assert type(MetaProduct) is Meta
    dynamic = Meta("Dynamic", (object,), {"answer": 42})
    assert dynamic.answer == 42
    assert type(dynamic) is Meta


def test_json_uses_ordinary_runtime_values():
    value = json.loads('{"small": 1, "large": 9223372036854775808, "items": [1, 2]}')
    assert value["small"] + value["large"] == 9223372036854775809
    assert value["items"] == [1, 2]
    assert json.loads(json.dumps(value))["large"] == 9223372036854775808


def test_class_attribute_falls_back_to_the_runtime_type():
    value = Child(1)
    assert value.__class__ is Child
    assert value.__class__.__name__ == "Child"
    assert (3).__class__ is int
    assert "x".__class__ is str
    assert json.__class__.__name__ == "module"


def test_range_is_a_type():
    values = range(1, 7, 2)
    assert isinstance(values, range)
    assert type(values) is range
    assert not isinstance([1, 3, 5], range)
    assert list(values) == [1, 3, 5]
    for arguments, message in [
        ((), "range expected at least 1 argument, got 0"),
        ((1, 2, 3, 4), "range expected at most 3 arguments, got 4"),
    ]:
        try:
            range(*arguments)
        except TypeError as error:
            assert str(error) == message
        else:
            raise AssertionError(arguments)


class ReadOnlyPoint:
    @property
    def x(self):
        return 1


def test_assigning_a_read_only_property_raises_attribute_error():
    # CPython names the class by its qualified name; a module-level class keeps them equal.
    try:
        ReadOnlyPoint().x = 2
    except AttributeError as error:
        assert str(error) == "property 'x' of 'ReadOnlyPoint' object has no setter"
    else:
        raise AssertionError("a property without a setter accepted a value")


class Half:
    """Equal to 0.5 and hashing like it, as a numeric type such as Fraction does."""

    def __eq__(self, other):
        return other == 0.5

    def __hash__(self):
        return hash(0.5)


class Never:
    def __eq__(self, other):
        return False

    __hash__ = object.__hash__


class Recorded:
    def __init__(self, value, log):
        self.value = value
        self.log = log

    def __eq__(self, other):
        self.log.append((self.value, other.value))
        return self.value == other.value

    __hash__ = None


def test_containers_compare_elements_with_eq():
    assert [Half()] == [0.5] and (0.5, Half()) == (Half(), 0.5)
    assert [[Half()], {"k": (Half(),)}] == [[0.5], {"k": (0.5,)}]
    assert [Half()] != [0.25] and not [Half()] == [0.5, 1]
    assert Half() in [0.5] and 0.5 in (Half(),) and Half() in {0.5} and Half() in {0.5: 1}
    assert {Half(): "a"}[0.5] == "a" and {0.5: "a"}.get(Half()) == "a"
    assert len({0.5: 1, Half(): 2}) == 1 and len({Half(), 0.5}) == 1
    assert {0.5: [Half()]} == {Half(): [0.5]} and {Half()} == {0.5}
    values = [1, Half(), 0.5]
    assert values.index(0.5) == 1 and values.count(0.5) == 2
    values.remove(0.5)
    assert len(values) == 2 and values[1] == 0.5


def test_container_equality_checks_identity_first_and_stops_at_the_first_difference():
    never = Never()
    assert [never] == [never] and never in [never] and never != never
    log = []
    left = [Recorded(1, log), Recorded(2, log), Recorded(3, log)]
    right = [Recorded(1, log), Recorded(5, log), Recorded(3, log)]
    assert left != right and log == [(1, 1), (2, 5)]
    first = [1]
    first.append(first)
    second = [1]
    second.append(second)
    assert first == first
    try:
        equal = first == second
    except RecursionError:
        pass
    else:
        raise AssertionError(f"comparing distinct self-containing lists returned {equal}")


def test_object_provides_identity_hash_and_equality():
    never = Never()
    assert hash(never) == object.__hash__(never) and {never: 1}[never] == 1
    assert object.__eq__(never, never) is True and object.__eq__(never, 1) is NotImplemented
    assert object.__ne__(never, 1) is True and object.__ne__(Half(), 0.5) is False
    assert object.__ne__(object(), 1) is NotImplemented


def test_builtin_iterators_expose_next_and_iter():
    iterator = iter([1, 2, 0.5, 4])
    assert iterator.__iter__() is iterator and iterator.__next__() == 1
    assert list(iter(iterator.__next__, Half())) == [2]
    remaining = iter(range(1))
    assert remaining.__next__() == 0
    try:
        remaining.__next__()
    except StopIteration:
        pass
    else:
        raise AssertionError("an exhausted iterator did not raise StopIteration")


def test_builtin_functions_and_bound_methods_have_names():
    import math

    functions = [round, len, print, sorted, math.floor, [].append, "".join, dict.fromkeys, object.__hash__]
    assert [function.__name__ for function in functions] == [
        "round",
        "len",
        "print",
        "sorted",
        "floor",
        "append",
        "join",
        "fromkeys",
        "__hash__",
    ]
    assert (repr(len), repr(math.floor)) == ("<built-in function len>", "<built-in function floor>")
    assert repr(list.append) == "<method 'append' of 'list' objects>"
    receiver = Base(1)
    method = receiver.describe
    assert method.__name__ == "describe" and method.__func__ is Base.describe
    assert method.__self__ is receiver


def test_getattr_runs_only_for_attributes_ordinary_lookup_misses():
    class Dynamic:
        real = "class attribute"

        def __init__(self):
            self.stored = "instance attribute"

        @property
        def computed(self):
            return "property"

        def __getattr__(self, name):
            if name.startswith("dyn_"):
                return name[4:]
            raise AttributeError(f"no {name} here")

    class Child(Dynamic):
        pass

    value = Dynamic()
    assert (value.real, value.stored, value.computed) == ("class attribute", "instance attribute", "property")
    assert value.dyn_x == "x" and value.dyn_y == "y" and Child().dyn_z == "z"
    assert hasattr(value, "dyn_w") and not hasattr(value, "other")
    assert getattr(value, "other", "default") == "default"
    try:
        missing = value.other
    except AttributeError as error:
        assert str(error) == "no other here"
    else:
        raise AssertionError(f"__getattr__'s AttributeError did not propagate: {missing!r}")

    class Broken:
        def __getattr__(self, name):
            raise KeyError(name)

    for probe in (lambda: Broken().x, lambda: hasattr(Broken(), "x")):
        try:
            probe()
        except KeyError:
            pass
        else:
            raise AssertionError("a non-AttributeError from __getattr__ must propagate")


def test_setattr_receives_every_assignment_and_object_setattr_stores():
    class Doubler:
        def __setattr__(self, name, value):
            object.__setattr__(self, name, value * 2)

    class Inherits(Doubler):
        pass

    doubled = Doubler()
    doubled.x = 2
    doubled.y = 5
    inherited = Inherits()
    inherited.z = 21
    assert (doubled.x, doubled.y, inherited.z) == (4, 10, 42)

    class Validated:
        def __init__(self):
            self.count = 0

        def __setattr__(self, name, value):
            if value < 0:
                raise ValueError("negative")
            super().__setattr__(name, value)

    validated = Validated()
    validated.count = 3
    try:
        validated.count = -1
    except ValueError:
        pass
    assert validated.count == 3

    class Frozen:
        def __init__(self, value):
            object.__setattr__(self, "value", value)

        def __setattr__(self, name, value):
            raise AttributeError(f"cannot assign to field '{name}'")

    frozen = Frozen(1)
    try:
        frozen.value = 2
    except AttributeError as error:
        assert str(error) == "cannot assign to field 'value'"
    assert frozen.value == 1

    class WithProperty:
        def __init__(self):
            self._x = 0

        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value + 100

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    with_property = WithProperty()
    with_property.x = 1
    assert with_property.x == 101
