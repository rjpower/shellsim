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
