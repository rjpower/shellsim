"""Portable exception semantics: the builtin hierarchy and the errors builtin operations raise.

Each expected message is CPython 3.14's text for the same operation.
"""


class AppError(ValueError):
    pass


class DetailedAppError(AppError):
    pass


def raised(operation):
    try:
        operation()
    except BaseException as error:
        return type(error), str(error)
    raise AssertionError("operation did not raise")


def test_builtin_hierarchy_matches_cpython():
    assert issubclass(KeyError, LookupError)
    assert issubclass(IndexError, LookupError)
    assert issubclass(ZeroDivisionError, ArithmeticError)
    assert issubclass(FloatingPointError, ArithmeticError)
    assert issubclass(ModuleNotFoundError, ImportError)
    assert issubclass(FileNotFoundError, OSError)
    assert issubclass(UnicodeDecodeError, ValueError)
    assert issubclass(RecursionError, RuntimeError)
    assert issubclass(NotImplementedError, RuntimeError)
    assert issubclass(RuntimeWarning, Warning)
    assert issubclass(UserWarning, Exception)
    assert issubclass(ValueError, object)
    assert not issubclass(SystemExit, Exception)
    assert not issubclass(KeyboardInterrupt, Exception)
    assert not issubclass(ValueError, LookupError)
    assert issubclass(KeyError, (TypeError, LookupError))


def test_user_exceptions_inherit_their_builtin_ancestors():
    assert issubclass(DetailedAppError, ValueError)
    assert issubclass(DetailedAppError, Exception)
    assert not issubclass(ValueError, AppError)
    assert not issubclass(int, ValueError)
    error = DetailedAppError("detail")
    assert isinstance(error, AppError)
    assert isinstance(error, ValueError)
    assert not isinstance(error, LookupError)
    try:
        raise error
    except ValueError as caught:
        assert caught is error


def test_except_clauses_match_builtin_ancestors():
    try:
        {}["missing"]
    except LookupError as error:
        assert isinstance(error, KeyError)
    else:
        raise AssertionError("missing key did not raise")
    try:
        divmod(1, 0)
    except ArithmeticError as error:
        assert type(error) is ZeroDivisionError
        assert type(error).__name__ == "ZeroDivisionError"
    else:
        raise AssertionError("division by zero did not raise")


# Operands live in variables so CPython does not warn about constant expressions at compile time.
NONE = None
FIVE = 5
EMPTY = []
ONE = [1]
LETTERS = "abc"

BUILTIN_OPERATION_ERRORS = [
    (lambda: EMPTY[1], IndexError, "list index out of range"),
    (lambda: (1,)[FIVE], IndexError, "tuple index out of range"),
    (lambda: LETTERS[FIVE], IndexError, "string index out of range"),
    (lambda: range(3)[FIVE], IndexError, "range object index out of range"),
    (lambda: [].pop(), IndexError, "pop from empty list"),
    (lambda: {}["x"], KeyError, "'x'"),
    (lambda: {}.pop("x"), KeyError, "'x'"),
    (lambda: set().remove(1), KeyError, "1"),
    (lambda: NONE + 1, TypeError, "unsupported operand type(s) for +: 'NoneType' and 'int'"),
    (lambda: -NONE, TypeError, "bad operand type for unary -: 'NoneType'"),
    (lambda: abs("a"), TypeError, "bad operand type for abs(): 'str'"),
    (lambda: FIVE < "a", TypeError, "'<' not supported between instances of 'int' and 'str'"),
    (lambda: ONE >= "a", TypeError, "'>=' not supported between instances of 'list' and 'str'"),
    (lambda: len(FIVE), TypeError, "object of type 'int' has no len()"),
    (lambda: iter(FIVE), TypeError, "'int' object is not iterable"),
    (lambda: FIVE[0], TypeError, "'int' object is not subscriptable"),
    (lambda: ONE["a"], TypeError, "list indices must be integers or slices, not str"),
    (lambda: LETTERS["a"], TypeError, "string indices must be integers, not 'str'"),
    (lambda: 1 in FIVE, TypeError, "argument of type 'int' is not a container or iterable"),
    (lambda: FIVE in LETTERS, TypeError, "'in <string>' requires string as left operand, not int"),
    (lambda: next(iter(EMPTY)), StopIteration, ""),
    (lambda: int("12x"), ValueError, "invalid literal for int() with base 10: '12x'"),
    (lambda: float("x"), ValueError, "could not convert string to float: 'x'"),
    (lambda: chr(-1), ValueError, "chr() arg not in range(0x110000)"),
    (lambda: range(0, 1, 0), ValueError, "range() arg 3 must not be zero"),
    (lambda: FIVE % 0, ZeroDivisionError, "division by zero"),
    (lambda: 2.0**10000, OverflowError, "(34, 'Numerical result out of range')"),
    (lambda: float(10**400), OverflowError, "int too large to convert to float"),
    (
        lambda: float(NONE),
        TypeError,
        "float() argument must be a string or a real number, not 'NoneType'",
    ),
]


def test_builtin_operations_raise_cpython_exceptions():
    for index, (operation, kind, message) in enumerate(BUILTIN_OPERATION_ERRORS):
        assert (index, *raised(operation)) == (index, kind, message)


def test_calling_a_non_callable_raises_type_error():
    value = 5
    assert raised(lambda: value()) == (TypeError, "'int' object is not callable")


def test_unpacking_reports_the_counts():
    def too_many():
        first, second = [1, 2, 3]
        return first, second

    def too_few():
        first, second, third = [1, 2]
        return first, second, third

    def starred():
        first, *rest, last = [1]
        return first, rest, last

    assert raised(too_many) == (ValueError, "too many values to unpack (expected 2, got 3)")
    assert raised(too_few) == (ValueError, "not enough values to unpack (expected 3, got 2)")
    assert raised(starred) == (ValueError, "not enough values to unpack (expected at least 2, got 1)")


def function(a, b=2, *, c):
    return a


def three(x, y, z):
    return x


def test_argument_binding_errors_are_type_errors():
    assert raised(lambda: function()) == (
        TypeError,
        "function() missing 1 required positional argument: 'a'",
    )
    assert raised(lambda: function(1)) == (
        TypeError,
        "function() missing 1 required keyword-only argument: 'c'",
    )
    assert raised(lambda: function(1, 2, 3, c=1)) == (
        TypeError,
        "function() takes from 1 to 2 positional arguments but 3 positional arguments "
        "(and 1 keyword-only argument) were given",
    )
    assert raised(lambda: function(1, d=1, c=1)) == (
        TypeError,
        "function() got an unexpected keyword argument 'd'",
    )
    assert raised(lambda: function(1, a=1, c=1)) == (
        TypeError,
        "function() got multiple values for argument 'a'",
    )
    assert raised(lambda: three()) == (
        TypeError,
        "three() missing 3 required positional arguments: 'x', 'y', and 'z'",
    )


def test_missing_names_and_attributes_raise():
    class Empty:
        pass

    assert raised(lambda: undefined_name) == (NameError, "name 'undefined_name' is not defined")  # noqa: F821
    assert raised(lambda: Empty().value) == (AttributeError, "'Empty' object has no attribute 'value'")
    assert raised(lambda: Empty.value) == (AttributeError, "type object 'Empty' has no attribute 'value'")
    import math

    assert raised(lambda: math.missing) == (AttributeError, "module 'math' has no attribute 'missing'")
    assert hasattr(Empty(), "value") is False
    assert getattr(Empty(), "value", 7) == 7


def test_imports_raise_module_not_found_and_import_error():
    try:
        import shellsim_missing_module  # noqa: F401
    except ModuleNotFoundError as error:
        assert str(error) == "No module named 'shellsim_missing_module'"
    else:
        raise AssertionError("import succeeded")
    try:
        from math import missing_name  # noqa: F401
    except ImportError as error:
        assert type(error) is ImportError
        assert str(error).startswith("cannot import name 'missing_name' from 'math'")
    else:
        raise AssertionError("import succeeded")
    from os import path

    assert path.join("a", "b") == "a/b"


def test_recursion_limit_raises_recursion_error():
    def recurse(depth):
        return recurse(depth + 1)

    assert raised(lambda: recurse(0)) == (RecursionError, "maximum recursion depth exceeded")


def test_ordering_comparisons_with_nan_are_false():
    nan = float("nan")
    assert not nan < 1
    assert not nan > 1
    assert not 1 <= nan
    assert not 1.5 >= nan
    assert not [nan] < [1]
    assert max(1, nan) == 1
    assert sorted([3, 1, 2]) == [1, 2, 3]


def test_membership_searches_any_iterable_lazily():
    seen = []

    def numbers():
        for number in range(10):
            seen.append(number)
            yield number

    assert 3 in numbers()
    assert seen == [0, 1, 2, 3]
    assert 4 not in iter([1, 2, 3])
    assert 2.0 in range(5)


def test_string_repr_chooses_quotes_like_cpython():
    assert repr("k") == "'k'"
    assert repr("'k'") == "\"'k'\""
    assert repr("a'b\"c") == "'a\\'b\"c'"
    assert repr("tab\there") == "'tab\\there'"
    assert repr("\x00\x7f") == "'\\x00\\x7f'"
    assert repr("é中") == "'é中'"
