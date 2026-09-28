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
        assert False, "slot wrapper accepted an unrelated receiver"


def test_builtin_sequence_iterators_are_exposed_as_descriptors():
    assert list([1, 2].__iter__()) == [1, 2]
    assert list(tuple.__iter__((3, 4))) == [3, 4]
    assert list(range(3).__iter__()) == [0, 1, 2]


def test_native_length_slots_are_directly_callable():
    assert list.__len__([1, 2]) == 2
    assert (1, 2, 3).__len__() == 3
    assert "café".__len__() == 4
    assert range(4).__len__() == 4
    assert {1, 2}.__len__() == 2
    assert dict.__len__({"x": 1}) == 1


def test_comparison_tries_subclass_reflection_first():
    class Left:
        def __lt__(self, other):
            return "left"

    class Right(Left):
        def __gt__(self, other):
            return "right"

    assert (Left() < Right()) == "right"


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
        assert False, "class defining equality kept an inherited hash"


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
        assert False, "direct default lookup called __getattr__"


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
        assert False, "inherited native getter accepted assignment"
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


def test_incompatible_builtin_payloads_raise_type_error():
    try:
        class Mixed(str, bytes):
            pass
    except TypeError:
        pass
    else:
        assert False, "incompatible builtin layouts were accepted"
