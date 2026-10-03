"""Names for built-in types that are not otherwise exposed as builtins."""

import sys as _sys


def _function():
    pass


class _Class:
    def _method(self):
        pass


def _generator():
    yield None


FunctionType = type(_function)
LambdaType = FunctionType
BuiltinFunctionType = type(len)
BuiltinMethodType = type([].append)
MethodType = type(_Class()._method)
ModuleType = type(_sys)
NoneType = type(None)
NotImplementedType = type(NotImplemented)
EllipsisType = type(Ellipsis)
GeneratorType = type(_generator())
MethodDescriptorType = type(str.join)
WrapperDescriptorType = type(object.__init__)
MethodWrapperType = type(object().__str__)
ClassMethodDescriptorType = type(dict.__dict__["fromkeys"]) if hasattr(dict, "__dict__") else classmethod
GetSetDescriptorType = type(FunctionType.__code__) if hasattr(FunctionType, "__code__") else property
MemberDescriptorType = GetSetDescriptorType
try:
    GenericAlias = type(list[int])
except TypeError:
    GenericAlias = None
try:
    UnionType = type(int | str)
except TypeError:
    UnionType = None
try:
    MappingProxyType = type(type.__dict__)
except AttributeError:
    MappingProxyType = dict
CellType = None
CodeType = None
FrameType = None
TracebackType = None
CoroutineType = None
AsyncGeneratorType = None
CapsuleType = None

del _function, _Class, _generator


class SimpleNamespace:
    """A simple attribute container with a readable repr and equality."""

    def __init__(self, mapping_or_iterable=(), **kwargs):
        if isinstance(mapping_or_iterable, dict):
            items = list(mapping_or_iterable.items())
        else:
            items = list(mapping_or_iterable)
        for key, value in items:
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        items = sorted(vars(self).items())
        body = ", ".join(key + "=" + repr(value) for key, value in items)
        name = "namespace" if type(self) is SimpleNamespace else type(self).__name__
        return name + "(" + body + ")"

    def __eq__(self, other):
        if isinstance(other, SimpleNamespace):
            return vars(self) == vars(other)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __replace__(self, **changes):
        result = type(self)(**vars(self))
        for key, value in changes.items():
            setattr(result, key, value)
        return result


class DynamicClassAttribute:
    """Route attribute access on an instance to a getter, on the class to ``__getattr__``.

    shellsim's ``property`` is a native constructor rather than a subclassable class, so this is
    a standalone descriptor with the same ``getter``/``setter``/``deleter`` surface.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc if doc is not None or fget is None else fget.__doc__
        self.overwrite_doc = doc is None
        self.__isabstractmethod__ = bool(getattr(fget, "__isabstractmethod__", False))

    def __get__(self, instance, ownerclass=None):
        if instance is None:
            if self.__isabstractmethod__:
                return self
            raise AttributeError()
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(instance)

    def getter(self, fget):
        doc = fget.__doc__ if self.overwrite_doc else self.__doc__
        result = type(self)(fget, self.fset, self.fdel, doc)
        result.overwrite_doc = self.overwrite_doc
        return result

    def setter(self, fset):
        result = type(self)(self.fget, fset, self.fdel, self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result

    def deleter(self, fdel):
        result = type(self)(self.fget, self.fset, fdel, self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result


def new_class(name, bases=(), kwds=None, exec_body=None):
    resolved_bases = resolve_bases(bases)
    meta, ns, kwds = prepare_class(name, resolved_bases, kwds)
    if exec_body is not None:
        exec_body(ns)
    if resolved_bases is not bases:
        ns["__orig_bases__"] = bases
    return meta(name, resolved_bases, ns, **kwds)


def resolve_bases(bases):
    new_bases = list(bases)
    updated = False
    shift = 0
    for index, base in enumerate(bases):
        if isinstance(base, type):
            continue
        if not hasattr(base, "__mro_entries__"):
            continue
        new_base = base.__mro_entries__(bases)
        updated = True
        if not isinstance(new_base, tuple):
            raise TypeError("__mro_entries__ must return a tuple")
        new_bases[index + shift : index + shift + 1] = new_base
        shift += len(new_base) - 1
    return tuple(new_bases) if updated else bases


def prepare_class(name, bases=(), kwds=None):
    kwds = {} if kwds is None else dict(kwds)
    if "metaclass" in kwds:
        meta = kwds.pop("metaclass")
    else:
        meta = type(bases[0]) if bases else type
    if isinstance(meta, type):
        meta = _calculate_meta(meta, bases)
    if hasattr(meta, "__prepare__"):
        ns = meta.__prepare__(name, bases, **kwds)
    else:
        ns = {}
    return meta, ns, kwds


def _calculate_meta(meta, bases):
    winner = meta
    for base in bases:
        base_meta = type(base)
        if issubclass(winner, base_meta):
            continue
        if issubclass(base_meta, winner):
            winner = base_meta
            continue
        raise TypeError("metaclass conflict")
    return winner


def get_original_bases(cls):
    try:
        return cls.__dict__.get("__orig_bases__", cls.__bases__)
    except AttributeError:
        raise TypeError("Expected an instance of type, not " + type(cls).__name__) from None


def coroutine(func):
    if not callable(func):
        raise TypeError("types.coroutine() expects a callable")
    return func
