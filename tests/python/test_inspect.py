# Portable checks of inspect.signature for Python callables.

import inspect
from inspect import Parameter, Signature, signature


def test_signature_renders_every_parameter_kind():
    def f(a, b=2, *args, c, d=None, **kw):
        pass

    def g(x, y, /, z=3, *, w):
        pass

    assert str(signature(f)) == "(a, b=2, *args, c, d=None, **kw)"
    assert str(signature(g)) == "(x, y, /, z=3, *, w)"
    assert str(signature(lambda: 0)) == "()"
    assert repr(signature(f)) == "<Signature (a, b=2, *args, c, d=None, **kw)>"
    kinds = [parameter.kind for parameter in signature(f).parameters.values()]
    assert kinds == [
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.VAR_POSITIONAL,
        Parameter.KEYWORD_ONLY,
        Parameter.KEYWORD_ONLY,
        Parameter.VAR_KEYWORD,
    ]
    assert [str(kind) for kind in kinds[:3]] == ["POSITIONAL_OR_KEYWORD"] * 2 + ["VAR_POSITIONAL"]
    assert Parameter.KEYWORD_ONLY == 3 and Parameter.VAR_KEYWORD > Parameter.KEYWORD_ONLY
    assert Parameter.KEYWORD_ONLY.description == "keyword-only"


def test_parameters_expose_defaults_and_empty_markers():
    def f(a, b=2):
        pass

    a, b = signature(f).parameters.values()
    assert a.name == "a" and a.default is Parameter.empty and a.annotation is inspect.Parameter.empty
    assert b.default == 2 and repr(b) == '<Parameter "b=2">'
    assert signature(f) == signature(f)
    built = Signature([Parameter("a", Parameter.POSITIONAL_ONLY), Parameter("b", Parameter.KEYWORD_ONLY, default=1)])
    assert str(built) == "(a, /, *, b=1)"


def test_methods_classes_and_callable_instances_omit_the_receiver():
    class Shape:
        def __init__(self, width, height=1):
            pass

        def scale(self, factor, *rest):
            pass

        def __call__(self, value):
            pass

    assert str(signature(Shape)) == "(width, height=1)"
    assert str(signature(Shape(2).scale)) == "(factor, *rest)"
    assert str(signature(Shape.scale)) == "(self, factor, *rest)"
    assert str(signature(Shape(2))) == "(value)"


def test_signature_rejects_bad_input():
    try:
        signature(5)
    except TypeError:
        pass
    else:
        raise AssertionError("signature accepted a non-callable")
    try:
        Parameter("x", Parameter.VAR_POSITIONAL, default=1)
    except ValueError as error:
        assert str(error) == "variadic positional parameters cannot have default values"
    else:
        raise AssertionError("a variadic parameter accepted a default")
