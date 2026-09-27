"""Signatures of Python callables: `signature`, `Signature`, and `Parameter`.

Parameter lists come from what the compiler recorded for each Python function, read through the
private `_shellsim_introspect` module. `signature` accepts Python functions, bound methods,
classes that define `__init__` in Python, and instances whose class defines `__call__` in
Python. Builtin and native callables have no recorded parameters and raise `ValueError`.
Annotations are not recorded, so every annotation is `Parameter.empty`. `Signature.bind` and
the rest of CPython's `inspect` module are not provided.
"""

from _shellsim_introspect import parameters as _recorded_parameters


class _empty:
    """Marker for a parameter without a default or annotation."""


class _ParameterKind:
    """One of the five parameter kinds, ordered and comparable like CPython's IntEnum."""

    def __init__(self, value, name, description):
        self.value = value
        self.name = name
        self.description = description

    def __repr__(self):
        return f"<_ParameterKind.{self.name}: {self.value}>"

    def __str__(self):
        return self.name

    def __int__(self):
        return self.value

    def __index__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == int(other) if isinstance(other, (int, _ParameterKind)) else NotImplemented

    def __lt__(self, other):
        return self.value < int(other)

    def __le__(self, other):
        return self.value <= int(other)

    def __gt__(self, other):
        return self.value > int(other)

    def __ge__(self, other):
        return self.value >= int(other)


_POSITIONAL_ONLY = _ParameterKind(0, "POSITIONAL_ONLY", "positional-only")
_POSITIONAL_OR_KEYWORD = _ParameterKind(1, "POSITIONAL_OR_KEYWORD", "positional or keyword")
_VAR_POSITIONAL = _ParameterKind(2, "VAR_POSITIONAL", "variadic positional")
_KEYWORD_ONLY = _ParameterKind(3, "KEYWORD_ONLY", "keyword-only")
_VAR_KEYWORD = _ParameterKind(4, "VAR_KEYWORD", "variadic keyword")
_KINDS = {
    kind.name: kind
    for kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD, _VAR_POSITIONAL, _KEYWORD_ONLY, _VAR_KEYWORD)
}


class Parameter:
    """One parameter of a signature: its `name`, `kind`, `default`, and `annotation`."""

    empty = _empty
    POSITIONAL_ONLY = _POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD = _POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL = _VAR_POSITIONAL
    KEYWORD_ONLY = _KEYWORD_ONLY
    VAR_KEYWORD = _VAR_KEYWORD

    def __init__(self, name, kind, *, default=_empty, annotation=_empty):
        if not isinstance(kind, _ParameterKind):
            raise ValueError(f"value {kind!r} is not a valid Parameter.kind")
        if default is not _empty and kind in (_VAR_POSITIONAL, _VAR_KEYWORD):
            raise ValueError(f"{kind.description} parameters cannot have default values")
        self.name = name
        self.kind = kind
        self.default = default
        self.annotation = annotation

    def __str__(self):
        if self.kind == _VAR_POSITIONAL:
            return "*" + self.name
        if self.kind == _VAR_KEYWORD:
            return "**" + self.name
        if self.default is _empty:
            return self.name
        return f"{self.name}={self.default!r}"

    def __repr__(self):
        return f'<Parameter "{self}">'

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return NotImplemented
        return (
            self.name == other.name
            and self.kind == other.kind
            and self.default == other.default
            and self.annotation == other.annotation
        )

    def __hash__(self):
        return hash((self.name, self.kind))


class Signature:
    """The ordered parameters of a callable, keyed by name in `parameters`."""

    empty = _empty

    def __init__(self, parameters=None, *, return_annotation=_empty):
        self.parameters = {}
        for parameter in parameters or ():
            if parameter.name in self.parameters:
                raise ValueError(f"duplicate parameter name: {parameter.name!r}")
            self.parameters[parameter.name] = parameter
        self.return_annotation = return_annotation

    def __str__(self):
        rendered = []
        slash_pending = False
        star_needed = True
        for parameter in self.parameters.values():
            kind = parameter.kind
            if kind == _POSITIONAL_ONLY:
                slash_pending = True
            elif slash_pending:
                rendered.append("/")
                slash_pending = False
            if kind == _VAR_POSITIONAL:
                star_needed = False
            elif kind == _KEYWORD_ONLY and star_needed:
                rendered.append("*")
                star_needed = False
            rendered.append(str(parameter))
        if slash_pending:
            rendered.append("/")
        return "(" + ", ".join(rendered) + ")"

    def __repr__(self):
        return f"<Signature {self}>"

    def __eq__(self, other):
        if not isinstance(other, Signature):
            return NotImplemented
        return (
            list(self.parameters.values()) == list(other.parameters.values())
            and self.return_annotation == other.return_annotation
        )

    def __hash__(self):
        return hash(tuple(self.parameters.values()))


def _signature_from(recorded, skip_first):
    parameters = [
        Parameter(name, _KINDS[kind], default=default if has_default else _empty)
        for name, kind, has_default, default in recorded
    ]
    return Signature(parameters[1:] if skip_first else parameters)


def signature(obj):
    """The `Signature` of a Python callable.

    ```python
    def f(a, b=2, *args, c, **kw): ...
    str(signature(f)) == "(a, b=2, *args, c, **kw)"
    ```
    """
    if not callable(obj):
        raise TypeError(f"{obj!r} is not a callable object")
    recorded = _recorded_parameters(obj)
    if recorded is not None:
        return _signature_from(recorded, False)
    # A class reports its `__init__` and a callable instance its class's `__call__`, both
    # without the receiver parameter.
    method = getattr(obj, "__init__", None) if isinstance(obj, type) else getattr(type(obj), "__call__", None)
    recorded = None if method is None else _recorded_parameters(method)
    if recorded is not None:
        return _signature_from(recorded, True)
    raise ValueError(f"no signature found for {obj!r}")
