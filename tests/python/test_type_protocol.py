"""Attribute lookup cases shared with CPython 3.14."""


def test_builtin_mro_uses_defining_namespace():
    assert bool.bit_length is int.bit_length
    assert True.bit_length() == 1
    assert bool.__add__ is int.__add__
    assert "__add__" not in bool.__dict__


def test_native_slot_wrappers_bind_and_keep_defining_implementation():
    assert (1).__add__(2) == 3
    assert int.__add__(1, 2) == 3
    assert int.__add__(1, "x") is NotImplemented

    class Doubled(int):
        def __add__(self, other):
            return 2 * super().__add__(other)

    assert Doubled(3) + 4 == 14
    assert int.__add__(Doubled(3), 4) == 7

    try:
        int.__add__("x", 1)
    except TypeError:
        pass
    else:
        raise AssertionError("slot wrapper accepted an unrelated receiver")


def test_builtin_sequence_iterators_are_exposed_as_descriptors():
    assert list([1, 2].__iter__()) == [1, 2]
    assert list(tuple.__iter__((3, 4))) == [3, 4]
    assert list(range(3).__iter__()) == [0, 1, 2]
    assert list(str.__iter__("éa")) == ["é", "a"]
    assert list(bytes.__iter__(b"ab")) == [97, 98]
    assert list(bytearray.__iter__(bytearray(b"ab"))) == [97, 98]
    assert sorted(set.__iter__({2, 1})) == [1, 2]
    assert sorted(frozenset.__iter__(frozenset({2, 1}))) == [1, 2]
    assert list(dict.__iter__({"a": 1, "b": 2})) == ["a", "b"]

    class Custom(str):
        def __iter__(self):
            return iter(["custom"])

    assert list(Custom("original")) == ["custom"]
    assert list(str.__iter__(Custom("original"))) == list("original")


def test_native_length_slots_are_directly_callable():
    assert list.__len__([1, 2]) == 2
    assert (1, 2, 3).__len__() == 3
    assert "café".__len__() == 4
    assert range(4).__len__() == 4
    assert {1, 2}.__len__() == 2
    assert dict.__len__({"x": 1}) == 1


def test_builtin_sequence_getitem_slots_keep_defining_implementation():
    assert list.__getitem__([4, 5], -1) == 5
    assert tuple.__getitem__((4, 5), 0) == 4
    assert str.__getitem__("café", 3) == "é"
    assert range.__getitem__(range(3), 1) == 1
    assert list.__getitem__([1, 2, 3], slice(1, None)) == [2, 3]

    class Doubled(tuple):
        def __getitem__(self, index):
            return 2 * super().__getitem__(index)

    values = Doubled((3, 4))
    assert values[1] == 8
    assert tuple.__getitem__(values, 1) == 4


def test_builtin_membership_slots_are_visible_and_keep_base_behavior():
    assert list.__contains__([1, 2], 2)
    assert tuple.__contains__((1, 2), 3) is False
    assert str.__contains__("café", "fé")
    assert bytes.__contains__(b"abc", 98)
    assert bytearray.__contains__(bytearray(b"abc"), 98)
    assert dict.__contains__({"a": 1}, "a")
    assert set.__contains__({1, 2}, 2)
    assert frozenset.__contains__(frozenset({1, 2}), 2)
    assert range.__contains__(range(4), 3)

    class Selective(list):
        def __contains__(self, item):
            return False

    value = Selective([1])
    assert 1 not in value
    assert list.__contains__(value, 1)


def test_format_uses_type_slot_and_exposes_builtin_descriptor():
    assert format(True) == "True"
    assert format(True, "d") == "1"
    assert format(3.5, ".2f") == "3.50"
    assert (3.5).__format__(".2f") == "3.50"
    assert str.__format__("hi", ">4") == "  hi"

    class Named:
        def __format__(self, spec):
            return "named:" + spec

    assert format(Named(), "x") == "named:x"

    class Plain:
        pass

    assert format(Plain()).startswith("<")
    try:
        format(Plain(), "x")
    except TypeError:
        pass
    else:
        raise AssertionError("object.__format__ accepted a non-empty spec")


def test_object_repr_and_str_are_directly_callable():
    class Custom:
        def __repr__(self):
            return "custom"

    value = Custom()
    assert "Custom object at 0x" in object.__repr__(value)
    assert object.__str__(value) == "custom"

    class Plain:
        pass

    plain = Plain()
    assert str(plain) == repr(plain)
    assert "Plain object at 0x" in str(plain)

    class Fancy:
        def __repr__(self):
            return "fancy " + super().__repr__()

    assert "Fancy object at 0x" in repr(Fancy())


def test_reversed_uses_slot_then_indexed_sequence():
    assert list(reversed([1, 2, 3])) == [3, 2, 1]
    assert list(list.__reversed__([1, 2])) == [2, 1]
    assert list(reversed("abc")) == ["c", "b", "a"]
    large = reversed(range(1_000_000_000))
    assert next(large) == 999_999_999
    assert next(large) == 999_999_998

    class Reversed:
        def __reversed__(self):
            return iter([7, 8])

    assert list(reversed(Reversed())) == [7, 8]

    class Sequence:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index * 2

    assert list(reversed(Sequence())) == [4, 2, 0]

    try:
        reversed(iter([1, 2]))
    except TypeError:
        pass
    else:
        raise AssertionError("reversed() accepted a bare iterator")


def test_class_subscription_and_builtin_generic_aliases():
    alias = list[int]
    assert alias.__origin__ is list
    assert alias.__args__ == (int,)
    assert repr(alias) == "list[int]"
    assert alias([1, 2]) == [1, 2]
    assert alias == list[int]
    assert hash(alias) == hash(list[int])
    assert dict[str, int].__args__ == (str, int)

    class Generic:
        def __class_getitem__(cls, item):
            return cls, item

    assert Generic[int] == (Generic, int)

    class Meta(type):
        def __getitem__(cls, item):
            return "metaclass"

    class Chosen(metaclass=Meta):
        def __class_getitem__(cls, item):
            return "class"

    assert Chosen[int] == "metaclass"

    class Numbers(list):
        pass

    assert Numbers[int].__origin__ is Numbers
    assert list.__class_getitem__(int) == list[int]


def test_set_name_runs_before_cooperative_init_subclass():
    events = []

    class Descriptor:
        def __set_name__(self, owner, name):
            events.append(("name", owner.__name__, name))

    class Parent:
        def __init_subclass__(cls):
            events.append(("parent", cls.__name__))
            super().__init_subclass__()

    class Child(Parent):
        field = Descriptor()

    assert events == [("name", "Child", "field"), ("parent", "Child")]


def test_type_call_runs_new_then_init_and_allows_metaclass_super():
    events = []

    class Meta(type):
        def __call__(cls, value):
            events.append("meta")
            return super().__call__(value)

    class Item(metaclass=Meta):
        def __new__(cls, value):
            events.append("new")
            return super().__new__(cls)

        def __init__(self, value):
            events.append("init")
            self.value = value

    assert Item(3).value == 3
    assert events == ["meta", "new", "init"]
    events.clear()
    assert type.__call__(Item, 4).value == 4
    assert events == ["new", "init"]
    assert type.__call__(list, [1, 2]) == [1, 2]

    class ReturnOther:
        def __new__(cls):
            return "other"

        def __init__(self):
            raise AssertionError("__init__ ran after __new__ returned another type")

    assert type.__call__(ReturnOther) == "other"


def test_default_constructor_uses_inherited_new_and_init():
    class Parent:
        def __init__(self, value):
            self.value = value

    class Child(Parent):
        pass

    assert Child(7).value == 7
    assert type.__call__(Child, 8).value == 8
    assert object.__new__(Child).__class__ is Child

    class Empty:
        pass

    assert type(Empty()) is Empty
    try:
        Empty(1)
    except TypeError:
        pass
    else:
        raise AssertionError("object constructor accepted an unused argument")


def test_comparison_tries_subclass_reflection_first():
    class Left:
        def __lt__(self, other):
            return "left"

    class Right(Left):
        def __gt__(self, other):
            return "right"

    assert (Left() < Right()) == "right"


def test_inplace_slot_precedes_binary_and_can_decline():
    class Inplace:
        def __iadd__(self, other):
            return "in-place"

        def __add__(self, other):
            return "binary"

    value = Inplace()
    value += 1
    assert value == "in-place"

    class Declining:
        def __iadd__(self, other):
            return NotImplemented

        def __add__(self, other):
            return "binary fallback"

    value = Declining()
    value += 1
    assert value == "binary fallback"


def test_delete_attribute_uses_type_slot():
    class Owner:
        def __init__(self):
            self.value = 3
            self.deleted = []

        def __delattr__(self, name):
            self.deleted.append(name)
            object.__delattr__(self, name)

    owner = Owner()
    del owner.value
    assert owner.deleted == ["value"]
    assert not hasattr(owner, "value")


def test_hash_rule_is_visible_in_class_namespace():
    class EqualOnly:
        def __eq__(self, other):
            return isinstance(other, EqualOnly)

    assert EqualOnly.__dict__["__hash__"] is None
    try:
        hash(EqualOnly())
    except TypeError:
        pass
    else:
        raise AssertionError("class defining equality kept an inherited hash")


def test_class_and_descriptor_mutation_after_cached_reads():
    class Base:
        pass

    value = Base()
    value.item = 1
    for _ in range(3):
        assert value.item == 1

    class Descriptor:
        def __get__(self, instance, owner):
            return 2

        def __set__(self, instance, new_value):
            pass

    Base.item = Descriptor()
    assert value.item == 2
    Descriptor.__get__ = lambda self, instance, owner: 3
    assert value.item == 3


def test_descriptor_attribute_error_uses_getattr_fallback():
    class Missing:
        def __get__(self, instance, owner):
            raise AttributeError("hidden")

    class Owner:
        value = Missing()

        def __getattr__(self, name):
            if name == "value":
                return 17
            raise AttributeError(name)

    assert Owner().value == 17
    assert not hasattr(Owner(), "absent")


def test_getattribute_override_and_direct_default_lookup():
    class Owner:
        def __getattribute__(self, name):
            if name == "value":
                return 11
            return object.__getattribute__(self, name)

        def __getattr__(self, name):
            return "fallback"

    owner = Owner()
    assert owner.value == 11
    assert owner.absent == "fallback"
    try:
        object.__getattribute__(owner, "absent")
    except AttributeError:
        pass
    else:
        raise AssertionError("direct default lookup called __getattr__")


def test_class_attribute_uses_inherited_object_descriptor():
    class Owner:
        pass

    owner = Owner()
    assert object.__getattribute__(owner, "__class__") is Owner
    assert Owner.__class__ is type
    assert list.__class__ is type
    assert (1).__class__ is int
    assert "__class__" in object.__dict__


def test_dictionary_descriptors_follow_type_and_instance_mros():
    class Parent:
        pass

    class Child(Parent):
        pass

    class Items(list):
        pass

    assert "__dict__" not in object.__dict__
    assert "__dict__" in type.__dict__
    assert "__dict__" in Parent.__dict__
    assert "__dict__" not in Child.__dict__
    assert "__dict__" in Items.__dict__
    assert not hasattr([], "__dict__")
    parent = Parent()
    parent.value = 3
    assert object.__getattribute__(parent, "__dict__") == {"value": 3}
    assert Child().__dict__ == {}


def test_type_metadata_descriptors_precede_class_namespace():
    class Parent:
        pass

    class Child(Parent):
        __name__ = "shadow"
        __bases__ = "shadow"
        __mro__ = "shadow"
        __module__ = "custom"

    assert Child.__name__ == "Child"
    assert Child.__dict__["__name__"] == "shadow"
    assert Child.__bases__ == (Parent,)
    assert Child.__mro__ == (Child, Parent, object)
    assert Child.__module__ == "custom"
    assert Child.mro() == [Child, Parent, object]


def test_registered_mro_orders_user_and_builtin_bases_together():
    class First:
        def __len__(self):
            return 7

    class Second(list):
        pass

    class Combined(First, Second):
        pass

    value = Combined([1])
    assert len(value) == 7
    assert Combined.__mro__ == (Combined, First, Second, list, object)


def test_metaclass_data_descriptor_precedes_class_namespace():
    class Meta(type):
        @property
        def value(cls):
            return "metaclass"

    class Owner(metaclass=Meta):
        value = "class"

    assert Owner.value == "metaclass"


def test_metaclass_getattribute_override():
    class Meta(type):
        def __getattribute__(cls, name):
            if name == "label":
                return "override"
            return type.__getattribute__(cls, name)

    class Owner(metaclass=Meta):
        label = "class"

    assert Owner.label == "override"
    assert type.__getattribute__(Owner, "label") == "class"


def test_metaclass_getattr_follows_raised_attribute_error():
    class Meta(type):
        def __getattribute__(cls, name):
            if name == "missing":
                raise AttributeError(name)
            return type.__getattribute__(cls, name)

        def __getattr__(cls, name):
            return "metaclass fallback"

    class Owner(metaclass=Meta):
        pass

    assert Owner.missing == "metaclass fallback"


def test_inherited_native_getter_rejects_assignment():
    class Integer(int):
        pass

    value = Integer(5)
    assert value.real == 5
    try:
        value.real = 99
    except AttributeError:
        pass
    else:
        raise AssertionError("inherited native getter accepted assignment")
    assert value.real == 5


def test_string_subclass_keeps_string_payload_and_methods():
    class Label(str):
        def upper_twice(self):
            return self.upper() * 2

    value = Label("café")
    assert type(value) is Label
    assert isinstance(value, str)
    assert value.upper_twice() == "CAFÉCAFÉ"
    assert value[0] == "c"
    assert len(value) == 4
    assert type(value + "!") is str
    assert hash(value) == hash("café")


def test_float_and_complex_subclasses_use_numeric_payloads():
    class Measure(float):
        pass

    class Coordinate(complex):
        pass

    measure = Measure("2.5")
    coordinate = Coordinate(2, 3)
    assert type(measure) is Measure
    assert type(coordinate) is Coordinate
    assert measure.real == 2.5
    assert coordinate.real == 2.0
    assert coordinate.imag == 3.0
    assert measure + 1 == 3.5
    assert coordinate + 1 == 3 + 3j
    assert type(measure + 1) is float
    assert type(coordinate + 1) is complex
    assert hash(measure) == hash(2.5)


def test_bytes_subclass_keeps_bytes_payload():
    class Packet(bytes):
        pass

    value = Packet(b"ab")
    assert type(value) is Packet
    assert isinstance(value, bytes)
    assert value[0] == 97
    assert len(value) == 2
    assert value.hex() == "6162"
    assert type(value + b"c") is bytes
    assert hash(value) == hash(b"ab")


def test_mutable_builtin_subclasses_keep_live_payloads():
    class Stack(list):
        def push(self, value):
            super().append(value)

    stack = Stack([1])
    stack.push(2)
    assert stack == [1, 2]
    assert len(stack) == 2
    assert type(stack + [3]) is list

    class Labels(set):
        pass

    labels = Labels(["a"])
    labels.add("b")
    assert labels == {"a", "b"}
    assert type(labels | {"c"}) is set

    class Buffer(bytearray):
        pass

    buffer = Buffer(b"a")
    buffer.append(98)
    assert buffer == bytearray(b"ab")
    assert type(buffer + b"c") is bytearray


def test_mutable_builtin_new_starts_empty_before_user_init():
    class Stack(list):
        def __init__(self, values):
            assert self == []
            super().__init__(values)

    class Labels(set):
        def __init__(self, values):
            assert self == set()
            super().__init__(values)

    class Buffer(bytearray):
        def __init__(self, values):
            assert self == bytearray()
            super().__init__(values)

    assert Stack([1, 2]) == [1, 2]
    assert Labels([1, 2]) == {1, 2}
    assert Buffer(b"ab") == bytearray(b"ab")


def test_frozenset_subclass_uses_builtin_payload():
    class Keys(frozenset):
        pass

    keys = Keys([1, 2])
    assert len(keys) == 2
    assert 1 in keys
    assert type(keys | {3}) is frozenset


def test_incompatible_builtin_payloads_raise_type_error():
    try:

        class Mixed(str, bytes):
            pass
    except TypeError:
        pass
    else:
        raise AssertionError("incompatible builtin layouts were accepted")
