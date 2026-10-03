"""Introspection of live objects: predicates, members, documentation and signatures.

Parameter lists come from what the compiler recorded for each Python function, read through the
private ``_shellsim_introspect`` module, which also reports whether a function is a generator or
coroutine function. Source text is not retained by the interpreter, so ``getsource`` and its
relatives raise ``OSError``. Annotations are not recorded, so every annotation is
``Parameter.empty``. Frames are not exposed: ``currentframe`` returns ``None`` and ``stack`` is
empty.
"""

import sys
import types
from _shellsim_introspect import parameters as _recorded_parameters, flags as _function_flags

__all__ = [
    "Parameter", "Signature", "BoundArguments", "signature", "isclass", "isfunction", "ismethod",
    "ismodule", "isbuiltin", "isroutine", "isgeneratorfunction", "iscoroutinefunction",
    "isasyncgenfunction", "isgenerator", "iscoroutine", "isawaitable", "isasyncgen", "isabstract",
    "ismethoddescriptor", "isdatadescriptor", "isgetsetdescriptor", "ismemberdescriptor",
    "istraceback", "isframe", "iscode", "getmembers", "getmembers_static", "getmodule", "getdoc",
    "cleandoc", "getcomments", "getfile", "getsourcefile", "getsource", "getsourcelines",
    "getmro", "unwrap", "getattr_static", "getfullargspec", "getcallargs", "get_annotations",
    "currentframe", "stack", "formatannotation", "FullArgSpec", "Attribute", "classify_class_attrs",
    "markcoroutinefunction", "CO_GENERATOR", "CO_COROUTINE", "CO_VARARGS", "CO_VARKEYWORDS",
    "CO_OPTIMIZED", "CO_NEWLOCALS", "CO_NESTED", "CO_ITERABLE_COROUTINE", "CO_ASYNC_GENERATOR",
]

CO_OPTIMIZED = 1
CO_NEWLOCALS = 2
CO_VARARGS = 4
CO_VARKEYWORDS = 8
CO_NESTED = 16
CO_GENERATOR = 32
CO_NOFREE = 64
CO_COROUTINE = 128
CO_ITERABLE_COROUTINE = 256
CO_ASYNC_GENERATOR = 512


# ---- predicates ----


def ismodule(object):
    return isinstance(object, types.ModuleType)


def isclass(object):
    return isinstance(object, type)


def ismethod(object):
    """A bound method: a function with a ``__self__``."""
    return isinstance(object, types.FunctionType) and hasattr(object, "__self__")


def isfunction(object):
    return isinstance(object, types.FunctionType) and not hasattr(object, "__self__") and _function_flags(object) is not None


def isbuiltin(object):
    """A native callable: one with no recorded Python parameter list."""
    return callable(object) and not isclass(object) and isinstance(object, types.FunctionType) and _function_flags(object) is None


def isroutine(object):
    return isfunction(object) or ismethod(object) or isbuiltin(object) or ismethoddescriptor(object)


def ismethoddescriptor(object):
    if isclass(object) or ismethod(object) or isfunction(object):
        return False
    if isinstance(object, (staticmethod, classmethod)):
        return True
    tp = type(object)
    return hasattr(tp, "__get__") and not hasattr(tp, "__set__")


def isdatadescriptor(object):
    if isclass(object) or ismethod(object) or isfunction(object):
        return False
    if isinstance(object, property):
        return True
    tp = type(object)
    return hasattr(tp, "__set__") or hasattr(tp, "__delete__")


def isgetsetdescriptor(object):
    return isinstance(object, types.GetSetDescriptorType) if types.GetSetDescriptorType is not None else False


def ismemberdescriptor(object):
    return isinstance(object, types.MemberDescriptorType) if types.MemberDescriptorType is not None else False


def istraceback(object):
    return types.TracebackType is not None and isinstance(object, types.TracebackType)


def isframe(object):
    return types.FrameType is not None and isinstance(object, types.FrameType)


def iscode(object):
    return types.CodeType is not None and isinstance(object, types.CodeType)


def _flags_of(object):
    object = unwrap(object) if hasattr(object, "__wrapped__") else object
    flags = _function_flags(object)
    if flags is None:
        func = getattr(object, "__func__", None)
        if func is not None:
            flags = _function_flags(func)
    return flags


def isgeneratorfunction(obj):
    flags = _flags_of(obj)
    return flags is not None and flags[0] and not flags[1]


def iscoroutinefunction(obj):
    if getattr(obj, "_is_coroutine_marker", None) is _is_coroutine_marker:
        return True
    flags = _flags_of(obj)
    return flags is not None and flags[1] and not flags[0]


def isasyncgenfunction(obj):
    flags = _flags_of(obj)
    return flags is not None and flags[0] and flags[1]


_is_coroutine_marker = object()


def markcoroutinefunction(func):
    func._is_coroutine_marker = _is_coroutine_marker
    return func


def _suspended_flags(object):
    """(is_generator, is_coroutine) of a generator, coroutine or async generator object."""
    if not isinstance(object, types.GeneratorType):
        return None
    return _function_flags(object)


def isgenerator(object):
    flags = _suspended_flags(object)
    return flags is not None and flags[0] and not flags[1]


def iscoroutine(object):
    flags = _suspended_flags(object)
    return flags is not None and flags[1] and not flags[0]


def isasyncgen(object):
    flags = _suspended_flags(object)
    return flags is not None and flags[0] and flags[1]


def isawaitable(object):
    return iscoroutine(object) or hasattr(type(object), "__await__")


def isabstract(object):
    if not isinstance(object, type):
        return False
    methods = getattr(object, "__abstractmethods__", None)
    return bool(methods)


# ---- members and documentation ----


def getmembers(object, predicate=None):
    """``(name, value)`` pairs of ``dir(object)``, sorted by name."""
    results = []
    for key in dir(object):
        try:
            value = getattr(object, key)
        except AttributeError:
            continue
        if predicate is None or predicate(value):
            results.append((key, value))
    results.sort(key=lambda pair: pair[0])
    return results


def getmembers_static(object, predicate=None):
    results = []
    for key in dir(object):
        try:
            value = getattr_static(object, key)
        except AttributeError:
            continue
        if predicate is None or predicate(value):
            results.append((key, value))
    results.sort(key=lambda pair: pair[0])
    return results


class Attribute(tuple):
    _fields = ("name", "kind", "defining_class", "object")

    def __new__(cls, name, kind, defining_class, object):
        return tuple.__new__(cls, (name, kind, defining_class, object))

    name = property(lambda self: self[0])
    kind = property(lambda self: self[1])
    defining_class = property(lambda self: self[2])
    object = property(lambda self: self[3])

    def __repr__(self):
        return "Attribute(name=%r, kind=%r, defining_class=%r, object=%r)" % tuple(self)


def classify_class_attrs(cls):
    """Each attribute of ``cls`` with its kind and the class in the MRO that defines it."""
    mro = getmro(cls)
    results = []
    names = dir(cls)
    for name in names:
        homecls = None
        obj = None
        for base in mro:
            namespace = getattr(base, "__dict__", {})
            if name in namespace:
                homecls = base
                obj = namespace[name]
                break
        if homecls is None:
            try:
                obj = getattr(cls, name)
            except AttributeError:
                continue
            homecls = cls
        if isinstance(obj, staticmethod):
            kind = "static method"
        elif isinstance(obj, classmethod):
            kind = "class method"
        elif isinstance(obj, property):
            kind = "property"
        elif isroutine(obj):
            kind = "method"
        else:
            kind = "data"
        results.append(Attribute(name, kind, homecls, obj))
    return results


def getmro(cls):
    return cls.__mro__


def getmodule(object, _filename=None):
    if ismodule(object):
        return object
    name = getattr(object, "__module__", None)
    if name is None:
        return None
    return sys.modules.get(name)


def getdoc(object):
    doc = getattr(object, "__doc__", None)
    if doc is None and isclass(object):
        for base in getmro(object)[1:]:
            doc = getattr(base, "__doc__", None)
            if doc is not None:
                break
    if not isinstance(doc, str):
        return None
    return cleandoc(doc)


def cleandoc(doc):
    """Strip the indentation docstrings carry from the source."""
    lines = doc.expandtabs().split("\n")
    margin = None
    for line in lines[1:]:
        content = len(line.lstrip(" "))
        if content:
            indent = len(line) - content
            margin = indent if margin is None else min(margin, indent)
    if lines:
        lines[0] = lines[0].lstrip(" ")
    if margin is not None:
        for index in range(1, len(lines)):
            lines[index] = lines[index][margin:]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    return "\n".join(lines)


def getcomments(object):
    return None


def getfile(object):
    if ismodule(object):
        file = getattr(object, "__file__", None)
        if file is not None:
            return file
        raise TypeError("%r is a built-in module" % object)
    if isclass(object):
        module = getmodule(object)
        if module is not None and hasattr(module, "__file__"):
            return module.__file__
        raise TypeError("%r is a built-in class" % object)
    if isfunction(object) or ismethod(object):
        module = getmodule(object)
        if module is not None and hasattr(module, "__file__"):
            return module.__file__
    raise TypeError(
        "module, class, method, function, traceback, frame, or code object was expected, got %s"
        % type(object).__name__
    )


def getsourcefile(object):
    return getfile(object)


def getsourcelines(object):
    raise OSError("source code is not retained by the interpreter")


def getsource(object):
    raise OSError("source code is not retained by the interpreter")


def unwrap(func, *, stop=None):
    """Follow ``__wrapped__`` to the original callable, detecting cycles."""
    if stop is None:
        def _is_wrapper(f):
            return hasattr(f, "__wrapped__")
    else:
        def _is_wrapper(f):
            return hasattr(f, "__wrapped__") and not stop(f)
    f = func
    memo = {id(f): f}
    recursion_limit = 1000
    while _is_wrapper(func):
        func = func.__wrapped__
        id_func = id(func)
        if id_func in memo or len(memo) >= recursion_limit:
            raise ValueError("wrapper loop when unwrapping {!r}".format(f))
        memo[id_func] = func
    return func


_sentinel = object()


def getattr_static(obj, attr, default=_sentinel):
    """Look ``attr`` up without triggering descriptors or ``__getattr__``."""
    instance_result = _sentinel
    if not isclass(obj):
        instance_dict = getattr(obj, "__dict__", None)
        if instance_dict is not None and attr in instance_dict:
            instance_result = instance_dict[attr]
        klass = type(obj)
    else:
        klass = obj
    klass_result = _sentinel
    for base in getmro(klass):
        namespace = getattr(base, "__dict__", None)
        if namespace is not None and attr in namespace:
            klass_result = namespace[attr]
            break
    if klass_result is not _sentinel and instance_result is not _sentinel:
        descriptor_type = type(klass_result)
        if hasattr(descriptor_type, "__get__") and (hasattr(descriptor_type, "__set__") or hasattr(descriptor_type, "__delete__")):
            return klass_result
    if instance_result is not _sentinel:
        return instance_result
    if klass_result is not _sentinel:
        return klass_result
    if isclass(obj):
        for base in getmro(type(obj)):
            namespace = getattr(base, "__dict__", None)
            if namespace is not None and attr in namespace:
                return namespace[attr]
    if default is not _sentinel:
        return default
    raise AttributeError(attr)


def get_annotations(obj, *, globals=None, locals=None, eval_str=False, format=None):
    annotations = getattr(obj, "__annotations__", None)
    if not isinstance(annotations, dict):
        return {}
    return dict(annotations)


def formatannotation(annotation, base_module=None):
    if getattr(annotation, "__module__", None) == "typing":
        return repr(annotation).replace("typing.", "")
    if isinstance(annotation, type):
        if annotation.__module__ in ("builtins", base_module):
            return annotation.__qualname__
        return annotation.__module__ + "." + annotation.__qualname__
    return repr(annotation)


def currentframe():
    return None


def stack(context=1):
    return []


# ---- signatures ----


class _empty:
    """Marker for a parameter without a default or annotation."""


class _void:
    """Marker for an argument that was not passed to ``Signature.replace``."""


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
        if not isinstance(name, str):
            raise TypeError("name must be a str, not a %s" % type(name).__name__)
        if not name.isidentifier():
            raise ValueError("{!r} is not a valid parameter name".format(name))
        self._name = name
        self._kind = kind
        self._default = default
        self._annotation = annotation

    @property
    def name(self):
        return self._name

    @property
    def kind(self):
        return self._kind

    @property
    def default(self):
        return self._default

    @property
    def annotation(self):
        return self._annotation

    def replace(self, *, name=_void, kind=_void, annotation=_void, default=_void):
        if name is _void:
            name = self._name
        if kind is _void:
            kind = self._kind
        if annotation is _void:
            annotation = self._annotation
        if default is _void:
            default = self._default
        return type(self)(name, kind, default=default, annotation=annotation)

    __replace__ = replace

    def __str__(self):
        formatted = self._name
        if self._annotation is not _empty:
            formatted = "{}: {}".format(formatted, formatannotation(self._annotation))
        if self._default is not _empty:
            if self._annotation is not _empty:
                formatted = "{} = {}".format(formatted, repr(self._default))
            else:
                formatted = "{}={}".format(formatted, repr(self._default))
        if self._kind == _VAR_POSITIONAL:
            formatted = "*" + formatted
        elif self._kind == _VAR_KEYWORD:
            formatted = "**" + formatted
        return formatted

    def __repr__(self):
        return f'<Parameter "{self}">'

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            return NotImplemented
        return (
            self._name == other._name
            and self._kind == other._kind
            and self._default == other._default
            and self._annotation == other._annotation
        )

    def __hash__(self):
        return hash((self._name, self._kind))


class BoundArguments:
    """The result of ``Signature.bind``: parameters mapped to argument values."""

    def __init__(self, signature, arguments):
        self.arguments = arguments
        self._signature = signature

    @property
    def signature(self):
        return self._signature

    @property
    def args(self):
        args = []
        for param_name, param in self._signature.parameters.items():
            if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                break
            try:
                arg = self.arguments[param_name]
            except KeyError:
                break
            if param.kind == _VAR_POSITIONAL:
                args.extend(arg)
            else:
                args.append(arg)
        return tuple(args)

    @property
    def kwargs(self):
        kwargs = {}
        kwargs_started = False
        for param_name, param in self._signature.parameters.items():
            if not kwargs_started:
                if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                    kwargs_started = True
                elif param_name not in self.arguments:
                    kwargs_started = True
                    continue
            if not kwargs_started:
                continue
            try:
                arg = self.arguments[param_name]
            except KeyError:
                pass
            else:
                if param.kind == _VAR_KEYWORD:
                    kwargs.update(arg)
                else:
                    kwargs[param_name] = arg
        return kwargs

    def apply_defaults(self):
        arguments = self.arguments
        new_arguments = []
        for name, param in self._signature.parameters.items():
            try:
                new_arguments.append((name, arguments[name]))
            except KeyError:
                if param.default is not _empty:
                    val = param.default
                elif param.kind is _VAR_POSITIONAL:
                    val = ()
                elif param.kind is _VAR_KEYWORD:
                    val = {}
                else:
                    continue
                new_arguments.append((name, val))
        self.arguments = dict(new_arguments)

    def __eq__(self, other):
        if not isinstance(other, BoundArguments):
            return NotImplemented
        return self.signature == other.signature and self.arguments == other.arguments

    def __repr__(self):
        args = ["{}={!r}".format(name, value) for name, value in self.arguments.items()]
        return "<{} ({})>".format(self.__class__.__name__, ", ".join(args))


class Signature:
    """The ordered parameters of a callable, keyed by name in `parameters`."""

    empty = _empty

    def __init__(self, parameters=None, *, return_annotation=_empty, __validate_parameters__=True):
        params = {}
        if parameters is not None:
            if __validate_parameters__:
                top_kind = _POSITIONAL_ONLY
                seen_default = False
                for param in parameters:
                    kind = param.kind
                    name = param.name
                    if kind < top_kind:
                        raise ValueError("wrong parameter order: {} parameter before {} parameter".format(top_kind.description, kind.description))
                    if kind > top_kind:
                        top_kind = kind
                    if kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD):
                        if param.default is _empty:
                            if seen_default:
                                raise ValueError("non-default argument follows default argument")
                        else:
                            seen_default = True
                    if name in params:
                        raise ValueError("duplicate parameter name: {!r}".format(name))
                    params[name] = param
            else:
                for param in parameters:
                    params[param.name] = param
        self._parameters = params
        self._return_annotation = return_annotation

    @property
    def parameters(self):
        return dict(self._parameters)

    @property
    def return_annotation(self):
        return self._return_annotation

    @classmethod
    def from_callable(cls, obj, *, follow_wrapped=True, globals=None, locals=None, eval_str=False):
        return _signature_from_callable(obj, follow_wrapped, cls)

    def replace(self, *, parameters=_void, return_annotation=_void):
        if parameters is _void:
            parameters = self._parameters.values()
        if return_annotation is _void:
            return_annotation = self._return_annotation
        return type(self)(parameters, return_annotation=return_annotation)

    __replace__ = replace

    def _bind(self, args, kwargs, *, partial=False):
        arguments = {}
        parameters = iter(self._parameters.values())
        parameters_ex = ()
        arg_vals = iter(args)
        while True:
            try:
                arg_val = next(arg_vals)
            except StopIteration:
                try:
                    param = next(parameters)
                except StopIteration:
                    break
                else:
                    if param.kind == _VAR_POSITIONAL:
                        break
                    if param.name in kwargs:
                        if param.kind == _POSITIONAL_ONLY:
                            if partial:
                                parameters_ex = (param,)
                                break
                            raise TypeError("missing a required positional-only argument: {arg!r}".format(arg=param.name))
                        parameters_ex = (param,)
                        break
                    if param.kind == _VAR_KEYWORD or param.default is not _empty:
                        parameters_ex = (param,)
                        break
                    if partial:
                        parameters_ex = (param,)
                        break
                    raise TypeError("missing a required argument: {arg!r}".format(arg=param.name))
            else:
                try:
                    param = next(parameters)
                except StopIteration:
                    raise TypeError("too many positional arguments") from None
                else:
                    if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                        raise TypeError("too many positional arguments")
                    if param.kind == _VAR_POSITIONAL:
                        values = [arg_val]
                        values.extend(arg_vals)
                        arguments[param.name] = tuple(values)
                        break
                    if param.name in kwargs and param.kind != _POSITIONAL_ONLY:
                        raise TypeError("multiple values for argument {arg!r}".format(arg=param.name))
                    arguments[param.name] = arg_val
        kwargs_param = None
        for param in list(parameters_ex) + list(parameters):
            if param.kind == _VAR_KEYWORD:
                kwargs_param = param
                continue
            if param.kind == _VAR_POSITIONAL:
                continue
            param_name = param.name
            try:
                arg_val = kwargs.pop(param_name)
            except KeyError:
                if not partial and param.kind != _VAR_POSITIONAL and param.default is _empty:
                    raise TypeError("missing a required argument: {arg!r}".format(arg=param_name)) from None
            else:
                if param.kind == _POSITIONAL_ONLY:
                    raise TypeError("{arg!r} parameter is positional only, but was passed as a keyword".format(arg=param.name))
                arguments[param_name] = arg_val
        if kwargs:
            if kwargs_param is not None:
                arguments[kwargs_param.name] = kwargs
            else:
                raise TypeError("got an unexpected keyword argument {arg!r}".format(arg=next(iter(kwargs))))
        return BoundArguments(self, arguments)

    def bind(self, /, *args, **kwargs):
        return self._bind(args, dict(kwargs))

    def bind_partial(self, /, *args, **kwargs):
        return self._bind(args, dict(kwargs), partial=True)

    def __str__(self):
        return self.format()

    def format(self, *, max_width=None):
        rendered = []
        slash_pending = False
        star_needed = True
        for parameter in self._parameters.values():
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
        text = "(" + ", ".join(rendered) + ")"
        if self._return_annotation is not _empty:
            text += " -> " + formatannotation(self._return_annotation)
        return text

    def __repr__(self):
        return "<Signature {}>".format(self)

    def __eq__(self, other):
        if not isinstance(other, Signature):
            return NotImplemented
        return (
            list(self._parameters.values()) == list(other._parameters.values())
            and self._return_annotation == other._return_annotation
        )

    def __hash__(self):
        return hash(tuple(self._parameters.values()))


def _signature_from(recorded, skip_first, cls=Signature):
    parameters = [
        Parameter(name, _KINDS[kind], default=default if has_default else _empty)
        for name, kind, has_default, default in recorded
    ]
    return cls(parameters[1:] if skip_first else parameters)


def _signature_from_callable(obj, follow_wrapped, cls):
    if not callable(obj):
        raise TypeError(f"{obj!r} is not a callable object")
    if follow_wrapped:
        obj = unwrap(obj, stop=lambda f: hasattr(f, "__signature__"))
    explicit = getattr(obj, "__signature__", None)
    if explicit is not None:
        if not isinstance(explicit, Signature):
            raise TypeError("unexpected object {!r} in __signature__ attribute".format(explicit))
        return explicit
    recorded = _recorded_parameters(obj)
    if recorded is not None:
        return _signature_from(recorded, False, cls)
    partial_func = getattr(obj, "func", None)
    partial_args = getattr(obj, "args", None)
    partial_keywords = getattr(obj, "keywords", None)
    if partial_func is not None and isinstance(partial_args, tuple) and isinstance(partial_keywords, dict):
        return _partial_signature(_signature_from_callable(partial_func, follow_wrapped, cls), partial_args, partial_keywords)
    # A class reports its `__init__` and a callable instance its class's `__call__`, both
    # without the receiver parameter.
    if isinstance(obj, type):
        for name in ("__init__", "__new__"):
            method = getattr(obj, name, None)
            recorded = None if method is None else _recorded_parameters(method)
            if recorded is not None:
                return _signature_from(recorded, True, cls)
        if obj is object or getattr(obj, "__init__", None) is object.__init__:
            return cls()
    else:
        method = getattr(type(obj), "__call__", None)
        recorded = None if method is None else _recorded_parameters(method)
        if recorded is not None:
            return _signature_from(recorded, True, cls)
    raise ValueError(f"no signature found for {obj!r}")


def _partial_signature(wrapped_sig, args, keywords):
    """The signature left after ``functools.partial`` fixes ``args`` and ``keywords``."""
    try:
        bound = wrapped_sig.bind_partial(*args, **keywords)
    except TypeError as exc:
        raise ValueError("partial object {!r} has incorrect arguments".format(args)) from exc
    new_params = dict(wrapped_sig.parameters)
    for name in bound.arguments:
        param = new_params[name]
        if name in keywords:
            if param.kind in (_POSITIONAL_OR_KEYWORD, _KEYWORD_ONLY):
                new_params[name] = param.replace(default=keywords[name], kind=_KEYWORD_ONLY)
            else:
                new_params.pop(name)
        else:
            if param.kind == _VAR_POSITIONAL:
                continue
            new_params.pop(name)
    for name, param in list(new_params.items()):
        if param.kind == _POSITIONAL_OR_KEYWORD and any(
            other.kind == _KEYWORD_ONLY and other.default is not _empty for other in new_params.values()
        ) and param.default is _empty:
            break
    return wrapped_sig.replace(parameters=list(new_params.values()))


def signature(obj, *, follow_wrapped=True, globals=None, locals=None, eval_str=False):
    """The `Signature` of a Python callable.

    ```python
    def f(a, b=2, *args, c, **kw): ...
    str(signature(f)) == "(a, b=2, *args, c, **kw)"
    ```
    """
    return Signature.from_callable(obj, follow_wrapped=follow_wrapped)


class FullArgSpec(tuple):
    _fields = ("args", "varargs", "varkw", "defaults", "kwonlyargs", "kwonlydefaults", "annotations")

    def __new__(cls, args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations):
        return tuple.__new__(cls, (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations))

    args = property(lambda self: self[0])
    varargs = property(lambda self: self[1])
    varkw = property(lambda self: self[2])
    defaults = property(lambda self: self[3])
    kwonlyargs = property(lambda self: self[4])
    kwonlydefaults = property(lambda self: self[5])
    annotations = property(lambda self: self[6])

    def __repr__(self):
        return ("FullArgSpec(args=%r, varargs=%r, varkw=%r, defaults=%r, kwonlyargs=%r, "
                "kwonlydefaults=%r, annotations=%r)") % tuple(self)


def getfullargspec(func):
    try:
        sig = Signature.from_callable(func, follow_wrapped=False)
    except Exception as exc:
        raise TypeError("unsupported callable") from exc
    args = []
    varargs = None
    varkw = None
    posonlyargs = []
    kwonlyargs = []
    defaults = ()
    kwdefaults = {}
    for param in sig.parameters.values():
        kind = param.kind
        name = param.name
        if kind == _POSITIONAL_ONLY:
            posonlyargs.append(name)
            if param.default is not _empty:
                defaults += (param.default,)
        elif kind == _POSITIONAL_OR_KEYWORD:
            args.append(name)
            if param.default is not _empty:
                defaults += (param.default,)
        elif kind == _VAR_POSITIONAL:
            varargs = name
        elif kind == _KEYWORD_ONLY:
            kwonlyargs.append(name)
            if param.default is not _empty:
                kwdefaults[name] = param.default
        elif kind == _VAR_KEYWORD:
            varkw = name
    if not kwdefaults:
        kwdefaults = None
    if not defaults:
        defaults = None
    return FullArgSpec(posonlyargs + args, varargs, varkw, defaults, kwonlyargs, kwdefaults, {})


def getcallargs(func, /, *positional, **named):
    """The arguments ``func(*positional, **named)`` would bind, with defaults applied."""
    sig = signature(func)
    bound = sig.bind(*positional, **named)
    bound.apply_defaults()
    return bound.arguments
