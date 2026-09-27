"""Portable ``operator`` semantics: every function follows the matching expression."""

import operator


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def __add__(self, other):
        return Vector(a + b for a, b in zip(self.values, other.values))

    def __iadd__(self, other):
        self.values = [a + b for a, b in zip(self.values, other.values)]
        return self


def test_arithmetic_and_comparison_functions_follow_the_operators():
    assert operator.add(2, 3) == 5
    assert operator.sub(2, 3) == -1
    assert operator.mul("ab", 2) == "abab"
    assert operator.truediv(7, 2) == 3.5
    assert operator.floordiv(-7, 2) == -4
    assert operator.mod(-7, 2) == 1
    assert operator.pow(2, 10) == 1024
    assert operator.neg(5) == -5
    assert operator.abs(-2.5) == 2.5
    assert operator.and_(6, 3) == 2
    assert operator.or_(6, 3) == 7
    assert operator.xor(6, 3) == 5
    assert operator.lshift(1, 4) == 16
    assert operator.invert(0) == -1
    assert operator.lt(1, 2) and operator.le(2, 2) and operator.ge(3, 2)
    assert operator.eq([1], [1]) and operator.ne(1, 2) and not operator.gt(1, 2)
    assert operator.not_([]) and operator.truth([0])
    assert operator.is_(None, None) and operator.is_not(1, None)
    assert operator.index(True) == 1


def test_sequence_functions():
    values = [1, 2, 3, 2]
    assert operator.contains(values, 3)
    assert operator.countOf(values, 2) == 2
    assert operator.indexOf(values, 2) == 1
    assert operator.getitem(values, 1) == 2
    operator.setitem(values, 0, 9)
    operator.delitem(values, -1)
    assert values == [9, 2, 3]
    assert operator.concat([1], [2]) == [1, 2]
    try:
        operator.concat(1, 2)
    except TypeError:
        pass
    else:
        raise AssertionError("concatenated integers")


def test_in_place_functions_mutate_and_return_the_target():
    values = [1]
    assert operator.iadd(values, [2]) is values
    assert values == [1, 2]
    vector = Vector([1, 2])
    assert operator.iadd(vector, Vector([10, 20])) is vector
    assert vector.values == [11, 22]
    assert operator.iadd(1, 2) == 3


def test_getters_and_method_callers():
    pairs = [(1, "b"), (0, "a")]
    assert sorted(pairs, key=operator.itemgetter(1)) == [(0, "a"), (1, "b")]
    assert operator.itemgetter(1, 0)("ab") == ("b", "a")
    assert operator.attrgetter("real")(3) == 3
    assert operator.attrgetter("values", "values")(Vector([1])) == ([1], [1])
    assert operator.methodcaller("replace", "a", "b")("aa") == "bb"
    assert repr(operator.itemgetter(1)) == "operator.itemgetter(1)"
