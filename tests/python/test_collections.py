"""Portable ``collections.namedtuple`` and ``typing.NamedTuple`` semantics."""

import keyword
import sys
from collections import namedtuple
from typing import NamedTuple

Point = namedtuple("Point", ["x", "y"])


class Employee(NamedTuple):
    name: str
    id: int = 3

    def badge(self):
        return f"{self.name}#{self.id}"


class Node(NamedTuple):
    value: int
    next: "Node | None" = None


class Labeled(Point):
    def total(self):
        return self.x + self.y


def error_of(operation):
    try:
        operation()
    except Exception as error:
        return type(error).__name__, str(error)
    raise AssertionError("expected an error")


def test_namedtuple_instances_are_tuples_with_named_fields():
    point = Point(1, y=2)
    assert (point.x, point[1], len(point), tuple(point)) == (1, 2, 2, (1, 2))
    assert point == (1, 2) and hash(point) == hash((1, 2)) and isinstance(point, tuple)
    assert repr(point) == "Point(x=1, y=2)" and repr(Point) == f"<class '{__name__}.Point'>"
    assert Point.__bases__ == (tuple,) and Point.__module__ == __name__
    assert Point._fields == ("x", "y") and Point.__match_args__ == ("x", "y")
    assert Point.__doc__ == "Point(x, y)" and namedtuple("One", "x").__doc__ == "One(x,)"
    assert point < Point(1, 3) and point + (9,) == (1, 2, 9) and point[::-1] == (2, 1)
    x, y = point
    assert (x, y, point.count(1), point.index(2)) == (1, 2, 1, 1)


def test_namedtuple_helpers_build_convert_and_replace():
    point = Point(1, 2)
    assert point._asdict() == {"x": 1, "y": 2} and type(point._asdict()) is dict
    assert point._replace(x=5) == Point(5, 2) and point.__replace__(y=0) == Point(1, 0)
    assert Point._make(iter([3, 4])) == Point(3, 4) and point.__getnewargs__() == (1, 2)
    assert error_of(lambda: Point._make([1, 2, 3])) == ("TypeError", "Expected 2 arguments, got 3")
    assert error_of(lambda: point._replace(z=1)) == ("TypeError", "Got unexpected field names: ['z']")
    assert error_of(lambda: setattr(point, "x", 3)) == ("AttributeError", "can't set attribute")
    assert error_of(lambda: delattr(point, "x")) == ("AttributeError", "can't delete attribute")
    assert Point.x.__doc__ == "Alias for field number 0"


def test_namedtuple_accepts_field_strings_defaults_rename_and_module():
    assert namedtuple("P", "a, b c")._fields == ("a", "b", "c")
    with_defaults = namedtuple("P", "x y z", defaults=[2, 3])
    assert with_defaults(1) == (1, 2, 3) and with_defaults._field_defaults == {"y": 2, "z": 3}
    renamed = namedtuple("P", "x class _y x 1z", rename=True)
    assert renamed._fields == ("x", "_1", "_2", "_3", "_4")
    assert namedtuple("P", "x", module="mymod").__module__ == "mymod"
    assert repr(namedtuple("Empty", "")()) == "Empty()"


def test_namedtuple_rejects_invalid_names_and_defaults():
    for arguments, message in [
        (("P", "x class"), "Type names and field names cannot be a keyword: 'class'"),
        (("P", "x 1y"), "Type names and field names must be valid identifiers: '1y'"),
        (("1P", "x"), "Type names and field names must be valid identifiers: '1P'"),
        (("P", "x _y"), "Field names cannot start with an underscore: '_y'"),
        (("P", "x x"), "Encountered duplicate field name: 'x'"),
    ]:
        assert error_of(lambda: namedtuple(*arguments)) == ("ValueError", message)
    too_many = error_of(lambda: namedtuple("P", "x", defaults=[1, 2]))
    assert too_many == ("TypeError", "Got more default values than field names")


def test_namedtuple_constructor_reports_argument_errors_like_a_signature():
    with_default = namedtuple("P", "x y z", defaults=[3])
    three = namedtuple("Q", "a b c")
    for operation, message in [
        (lambda: Point(1), "Point.__new__() missing 1 required positional argument: 'y'"),
        (lambda: Point(), "Point.__new__() missing 2 required positional arguments: 'x' and 'y'"),
        (lambda: three(), "Q.__new__() missing 3 required positional arguments: 'a', 'b', and 'c'"),
        (lambda: Point(1, 2, 3), "Point.__new__() takes 3 positional arguments but 4 were given"),
        (
            lambda: with_default(1, 2, 3, 4),
            "P.__new__() takes from 3 to 4 positional arguments but 5 were given",
        ),
        (lambda: Point(1, 2, x=3), "Point.__new__() got multiple values for argument 'x'"),
        (lambda: Point(1, z=3), "Point.__new__() got an unexpected keyword argument 'z'"),
        (lambda: with_default(1, 2, 3, 4, z=1), "P.__new__() got multiple values for argument 'z'"),
    ]:
        assert error_of(operation) == ("TypeError", message)


def test_namedtuple_subclasses_keep_their_name_and_methods():
    labeled = Labeled(3, 4)
    assert (repr(labeled), labeled.total()) == ("Labeled(x=3, y=4)", 7)
    assert type(labeled._replace(x=0)) is Labeled and Labeled._make([1, 1]) == (1, 1)


def test_typing_named_tuple_classes_use_annotations_and_defaults():
    employee = Employee("a")
    assert (employee, employee.badge()) == (("a", 3), "a#3")
    assert repr(employee) == "Employee(name='a', id=3)"
    assert Employee._fields == ("name", "id") and Employee._field_defaults == {"id": 3}
    assert Employee.__bases__ == (tuple,) and Employee.__module__ == __name__
    assert Node(1, Node(2)).next.value == 2 and Node(1)._asdict() == {"value": 1, "next": None}


def test_typing_named_tuple_function_and_class_errors():
    Pair = NamedTuple("Pair", [("a", int), ("b", int)])
    assert (Pair(3, 4), Pair.__module__) == ((3, 4), __name__)
    unpack = error_of(lambda: NamedTuple("E", [("name",)]))
    assert unpack == ("ValueError", "not enough values to unpack (expected 2, got 1)")

    def non_default_after_default():
        class Broken(NamedTuple):
            a: int = 1
            b: int = 2
            c: int

    def overwrite():
        class Broken(NamedTuple):
            x: int

            def _make(self):
                pass

    def two_bases():
        class Broken(NamedTuple, object):
            x: int

    assert error_of(non_default_after_default) == (
        "TypeError",
        "Non-default namedtuple field c cannot follow default fields a, b",
    )
    assert error_of(overwrite) == ("AttributeError", "Cannot overwrite NamedTuple attribute _make")
    assert error_of(two_bases) == ("TypeError", "can only inherit from a NamedTuple type and Generic")


def test_keyword_and_identifier_predicates():
    assert keyword.iskeyword("class") and not keyword.iskeyword("match")
    assert keyword.issoftkeyword("match") and "type" in keyword.softkwlist
    assert not keyword.iskeyword(1) and len(keyword.kwlist) == 35
    assert ["a1".isidentifier(), "1a".isidentifier(), "_".isidentifier(), "".isidentifier()] == [
        True,
        False,
        True,
        False,
    ]
    assert "é".isidentifier() and "a·b".isidentifier() and not "·a".isidentifier()


def test_getframemodulename_reports_the_calling_module():
    def caller():
        return sys._getframemodulename(), sys._getframemodulename(depth=1)

    assert caller() == (__name__, __name__)
    assert sys._getframemodulename(100) is None
