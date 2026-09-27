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
