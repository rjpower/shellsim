"""Abstract base classes.

``ABCMeta`` records the abstract names of each class it creates in ``__abstractmethods__``,
which the interpreter checks when the class is instantiated. ``isinstance`` and ``issubclass``
consult ``__subclasshook__``, classes passed to ``register`` and the ABC subclasses the
metaclass tracks itself, since the interpreter has no ``__subclasses__``.
"""

_cache_token = 0


def get_cache_token():
    """A value that changes whenever an ABC's registry changes, for cache invalidation."""
    return _cache_token


def abstractmethod(funcobj):
    funcobj.__isabstractmethod__ = True
    return funcobj


class abstractclassmethod:
    """A class method that must be overridden; prefer ``classmethod`` over ``abstractmethod``."""

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        self.__func__ = callable

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        function = self.__func__

        def bound(*args, **kwargs):
            return function(owner, *args, **kwargs)

        bound.__isabstractmethod__ = True
        return bound


class abstractstaticmethod:
    """A static method that must be overridden; prefer ``staticmethod`` over ``abstractmethod``."""

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        self.__func__ = callable

    def __get__(self, instance, owner=None):
        return self.__func__


class abstractproperty:
    """A property that must be overridden; prefer ``property`` over ``abstractmethod``."""

    __isabstractmethod__ = True

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc if doc is not None or fget is None else fget.__doc__

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
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
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


def _is_abstract(value):
    return bool(getattr(value, "__isabstractmethod__", False))


def _compute_abstract_methods(cls, namespace, bases):
    abstracts = set()
    for name, value in namespace.items():
        if _is_abstract(value):
            abstracts.add(name)
    for base in bases:
        for name in getattr(base, "__abstractmethods__", ()):
            if _is_abstract(getattr(cls, name, None)):
                abstracts.add(name)
    return frozenset(abstracts)


class ABCMeta(type):
    """Metaclass for abstract base classes.

    ```python
    class Shape(ABC):
        @abstractmethod
        def area(self): ...

    Shape()            # TypeError: Can't instantiate abstract class Shape ...
    Shape.register(Square)
    issubclass(Square, Shape)  # True
    ```
    """

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls.__abstractmethods__ = _compute_abstract_methods(cls, namespace, bases)
        cls._abc_registry = []
        cls._abc_subclasses = []
        cls._abc_cache = set()
        cls._abc_negative_cache = set()
        cls._abc_negative_cache_version = _cache_token
        for base in bases:
            if isinstance(base, ABCMeta):
                base._abc_subclasses.append(cls)
        return cls

    def register(cls, subclass):
        """Register ``subclass`` as a virtual subclass of this ABC and return it."""
        global _cache_token
        if not isinstance(subclass, type):
            raise TypeError("Can only register classes")
        if issubclass(subclass, cls):
            return subclass
        if issubclass(cls, subclass):
            raise RuntimeError("Refusing to create an inheritance cycle")
        cls._abc_registry.append(subclass)
        _cache_token += 1
        return subclass

    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        if not isinstance(subclass, type):
            raise TypeError("issubclass() arg 1 must be a class")
        if subclass in cls._abc_cache:
            return True
        if cls._abc_negative_cache_version < _cache_token:
            cls._abc_negative_cache = set()
            cls._abc_negative_cache_version = _cache_token
        elif subclass in cls._abc_negative_cache:
            return False
        hook = getattr(cls, "__subclasshook__", None)
        answer = hook(subclass) if hook is not None else NotImplemented
        if answer is NotImplemented:
            answer = cls in getattr(subclass, "__mro__", ())
            if not answer:
                answer = any(issubclass(subclass, registered) for registered in cls._abc_registry)
            if not answer:
                answer = any(issubclass(subclass, child) for child in cls._abc_subclasses)
        if answer:
            cls._abc_cache.add(subclass)
            return True
        cls._abc_negative_cache.add(subclass)
        return False

    def _dump_registry(cls, file=None):
        print("Class: %s.%s" % (cls.__module__, cls.__qualname__), file=file)
        print("Inv. counter: %s" % _cache_token, file=file)
        print("_abc_registry: %r" % (cls._abc_registry,), file=file)
        print("_abc_cache: %r" % (cls._abc_cache,), file=file)
        print("_abc_negative_cache: %r" % (cls._abc_negative_cache,), file=file)

    def _abc_registry_clear(cls):
        cls._abc_registry = []

    def _abc_caches_clear(cls):
        cls._abc_cache = set()
        cls._abc_negative_cache = set()


def update_abstractmethods(cls):
    """Recompute ``__abstractmethods__`` after methods were added to ``cls``."""
    if not hasattr(cls, "__abstractmethods__"):
        return cls
    abstracts = set()
    for base in cls.__bases__:
        for name in getattr(base, "__abstractmethods__", ()):
            if _is_abstract(getattr(cls, name, None)):
                abstracts.add(name)
    for name, value in cls.__dict__.items():
        if _is_abstract(value):
            abstracts.add(name)
    cls.__abstractmethods__ = frozenset(abstracts)
    return cls


class ABC(metaclass=ABCMeta):
    """Helper class that creates an ABC through inheritance."""

    __slots__ = ()
