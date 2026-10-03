"""Higher-order functions and operations on callables.

``reduce`` is native; everything else is written over ordinary Python protocols. Caches are
plain dicts, so their cost is the modeled memory of the entries they hold.
"""

from _functools import reduce

__all__ = [
    "update_wrapper", "wraps", "WRAPPER_ASSIGNMENTS", "WRAPPER_UPDATES", "total_ordering",
    "cache", "cmp_to_key", "lru_cache", "reduce", "partial", "partialmethod", "Placeholder",
    "singledispatch", "singledispatchmethod", "cached_property", "recursive_repr",
]

WRAPPER_ASSIGNMENTS = ("__module__", "__name__", "__qualname__", "__doc__", "__type_params__")
WRAPPER_UPDATES = ("__dict__",)


def update_wrapper(wrapper, wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES):
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            continue
        try:
            setattr(wrapper, attr, value)
        except (AttributeError, TypeError):
            pass
    for attr in updated:
        source = getattr(wrapped, attr, None)
        if source:
            for key, value in dict(source).items():
                try:
                    setattr(wrapper, key, value)
                except (AttributeError, TypeError):
                    pass
    wrapper.__wrapped__ = wrapped
    return wrapper


def wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES):
    def decorator(wrapper):
        return update_wrapper(wrapper, wrapped, assigned, updated)

    return decorator


class _PlaceholderType:
    """Sentinel marking a positional slot a later ``partial`` call fills."""

    __slots__ = ()

    def __repr__(self):
        return "Placeholder"

    def __reduce__(self):
        return "Placeholder"


Placeholder = _PlaceholderType()


class partial:
    """A callable with some positional and keyword arguments fixed in advance."""

    def __init__(self, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")
        if args and args[-1] is Placeholder:
            raise TypeError("trailing Placeholders are not allowed")
        if isinstance(func, partial):
            keywords = {**func.keywords, **keywords}
            args = _merge_placeholders(func.args, args)
            func = func.func
        self.func = func
        self.args = args
        self.keywords = keywords

    def __call__(self, /, *args, **keywords):
        merged = {**self.keywords, **keywords}
        if Placeholder in self.args:
            positional = list(self.args)
            needed = positional.count(Placeholder)
            if len(args) < needed:
                raise TypeError(
                    "missing positional arguments in 'partial' call; expected at least %d, got %d"
                    % (needed, len(args))
                )
            supplied = iter(args)
            for index, value in enumerate(positional):
                if value is Placeholder:
                    positional[index] = next(supplied)
            return self.func(*positional, *supplied, **merged)
        return self.func(*self.args, *args, **merged)

    def __repr__(self):
        parts = [repr(self.func)]
        parts.extend(repr(value) for value in self.args)
        parts.extend("%s=%r" % item for item in self.keywords.items())
        return "functools.partial(" + ", ".join(parts) + ")"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return partial(self, instance)


def _merge_placeholders(existing, new):
    """Fill the placeholders of ``existing`` from ``new`` and append the rest."""
    if Placeholder not in existing:
        return existing + new
    merged = list(existing)
    supplied = iter(new)
    for index, value in enumerate(merged):
        if value is Placeholder:
            try:
                merged[index] = next(supplied)
            except StopIteration:
                break
    return tuple(merged) + tuple(supplied)


class partialmethod:
    """A ``partial`` for methods: the receiver is prepended when the attribute is accessed."""

    def __init__(self, func, /, *args, **keywords):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError("%r is not callable or a descriptor" % func)
        if isinstance(func, partialmethod):
            self.func = func.func
            self.args = func.args + args
            self.keywords = {**func.keywords, **keywords}
        else:
            self.func = func
            self.args = args
            self.keywords = keywords

    def __repr__(self):
        parts = [repr(self.func)]
        parts.extend(repr(value) for value in self.args)
        parts.extend("%s=%r" % item for item in self.keywords.items())
        return "functools.partialmethod(" + ", ".join(parts) + ")"

    def _make_unbound_method(self):
        def method(cls_or_self, /, *args, **keywords):
            return self.func(cls_or_self, *self.args, *args, **{**self.keywords, **keywords})

        method.__isabstractmethod__ = self.__isabstractmethod__
        method._partialmethod = self
        return method

    def __get__(self, instance, owner=None):
        get = getattr(self.func, "__get__", None)
        if get is not None:
            bound = get(instance, owner)
            if bound is not self.func:
                result = partial(bound, *self.args, **self.keywords)
                return result
        if instance is None:
            return self._make_unbound_method()
        return partial(self.func, instance, *self.args, **self.keywords)

    @property
    def __isabstractmethod__(self):
        return bool(getattr(self.func, "__isabstractmethod__", False))


def cmp_to_key(mycmp):
    """Convert a ``cmp(a, b)`` function into a key function for sorting."""

    class K:
        __slots__ = ["obj"]

        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        __hash__ = None

    return K


def _gt_from_lt(self, other):
    result = type(self).__lt__(self, other)
    if result is NotImplemented:
        return result
    return not result and self != other


def _le_from_lt(self, other):
    result = type(self).__lt__(self, other)
    if result is NotImplemented:
        return result
    return result or self == other


def _ge_from_lt(self, other):
    result = type(self).__lt__(self, other)
    if result is NotImplemented:
        return result
    return not result


def _ge_from_le(self, other):
    result = type(self).__le__(self, other)
    if result is NotImplemented:
        return result
    return not result or self == other


def _lt_from_le(self, other):
    result = type(self).__le__(self, other)
    if result is NotImplemented:
        return result
    return result and self != other


def _gt_from_le(self, other):
    result = type(self).__le__(self, other)
    if result is NotImplemented:
        return result
    return not result


def _lt_from_gt(self, other):
    result = type(self).__gt__(self, other)
    if result is NotImplemented:
        return result
    return not result and self != other


def _ge_from_gt(self, other):
    result = type(self).__gt__(self, other)
    if result is NotImplemented:
        return result
    return result or self == other


def _le_from_gt(self, other):
    result = type(self).__gt__(self, other)
    if result is NotImplemented:
        return result
    return not result


def _le_from_ge(self, other):
    result = type(self).__ge__(self, other)
    if result is NotImplemented:
        return result
    return not result or self == other


def _gt_from_ge(self, other):
    result = type(self).__ge__(self, other)
    if result is NotImplemented:
        return result
    return result and self != other


def _lt_from_ge(self, other):
    result = type(self).__ge__(self, other)
    if result is NotImplemented:
        return result
    return not result


_convert = {
    "__lt__": [("__gt__", _gt_from_lt), ("__le__", _le_from_lt), ("__ge__", _ge_from_lt)],
    "__le__": [("__ge__", _ge_from_le), ("__lt__", _lt_from_le), ("__gt__", _gt_from_le)],
    "__gt__": [("__lt__", _lt_from_gt), ("__ge__", _ge_from_gt), ("__le__", _le_from_gt)],
    "__ge__": [("__le__", _le_from_ge), ("__gt__", _gt_from_ge), ("__lt__", _lt_from_ge)],
}


def _defines(cls, name):
    for base in cls.__mro__:
        if base is object:
            return False
        if name in getattr(base, "__dict__", {}):
            return True
    return False


def total_ordering(cls):
    """Fill in the missing ordering methods from the one the class defines."""
    roots = [op for op in _convert if _defines(cls, op)]
    if not roots:
        raise ValueError("must define at least one ordering operation: < > <= >=")
    root = max(roots)
    for opname, opfunc in _convert[root]:
        if opname not in roots:
            opfunc.__name__ = opname
            setattr(cls, opname, opfunc)
    return cls


class _CacheInfo(tuple):
    _fields = ("hits", "misses", "maxsize", "currsize")

    def __new__(cls, hits, misses, maxsize, currsize):
        return tuple.__new__(cls, (hits, misses, maxsize, currsize))

    hits = property(lambda self: self[0])
    misses = property(lambda self: self[1])
    maxsize = property(lambda self: self[2])
    currsize = property(lambda self: self[3])

    def __repr__(self):
        return "CacheInfo(hits=%r, misses=%r, maxsize=%r, currsize=%r)" % tuple(self)


class _HashedSeq(list):
    __slots__ = ("hashvalue",)

    def __init__(self, tup):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


_kwd_mark = object()


def _make_key(args, kwds, typed):
    key = args
    if kwds:
        key += (_kwd_mark,)
        for item in kwds.items():
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    elif len(key) == 1 and type(key[0]) in (int, str):
        return key[0]
    return key


def lru_cache(maxsize=128, typed=False):
    """Memoize a function on its arguments, keeping the ``maxsize`` most recent results."""
    if isinstance(maxsize, int):
        if maxsize < 0:
            maxsize = 0
    elif callable(maxsize) and isinstance(typed, bool):
        user_function, maxsize = maxsize, 128
        return _lru_cache_wrapper(user_function, maxsize, typed)
    elif maxsize is not None:
        raise TypeError("Expected first argument to be an integer, a callable, or None")

    def decorating_function(user_function):
        return _lru_cache_wrapper(user_function, maxsize, typed)

    return decorating_function


def _lru_cache_wrapper(user_function, maxsize, typed):
    cache = {}
    stats = [0, 0]

    if maxsize == 0:

        def wrapper(*args, **kwds):
            stats[1] += 1
            return user_function(*args, **kwds)

    elif maxsize is None:

        def wrapper(*args, **kwds):
            key = _make_key(args, kwds, typed)
            if key in cache:
                stats[0] += 1
                return cache[key]
            stats[1] += 1
            result = user_function(*args, **kwds)
            cache[key] = result
            return result

    else:

        def wrapper(*args, **kwds):
            key = _make_key(args, kwds, typed)
            if key in cache:
                stats[0] += 1
                result = cache.pop(key)
                cache[key] = result
                return result
            stats[1] += 1
            result = user_function(*args, **kwds)
            if key in cache:
                pass
            elif len(cache) >= maxsize:
                del cache[next(iter(cache))]
            cache[key] = result
            return result

    def cache_info():
        return _CacheInfo(stats[0], stats[1], maxsize, len(cache))

    def cache_clear():
        cache.clear()
        stats[0] = stats[1] = 0

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    wrapper.cache_parameters = lambda: {"maxsize": maxsize, "typed": typed}
    return update_wrapper(wrapper, user_function)


def cache(user_function, /):
    """An unbounded ``lru_cache``."""
    return lru_cache(maxsize=None)(user_function)


class singledispatch:
    """A function whose implementation is chosen by the type of its first argument.

    ``register`` takes the class explicitly: ``@fun.register(int)``. Registering through type
    annotations is not supported because the interpreter does not record annotations.
    """

    def __init__(self, func):
        self.registry = {object: func}
        self._cache = {}
        self.func = func
        update_wrapper(self, func)

    def dispatch(self, cls):
        try:
            return self._cache[cls]
        except KeyError:
            pass
        for candidate in getattr(cls, "__mro__", (cls, object)):
            if candidate in self.registry:
                implementation = self.registry[candidate]
                break
        else:
            implementation = self._match_abstract(cls)
        self._cache[cls] = implementation
        return implementation

    def _match_abstract(self, cls):
        for registered, implementation in self.registry.items():
            if registered is not object and issubclass(cls, registered):
                return implementation
        return self.registry[object]

    def register(self, cls, func=None):
        if func is None:
            if isinstance(cls, type):
                return lambda f: self.register(cls, f)
            raise TypeError(
                "Invalid first argument to `register()`: %r. Use either `@register(some_class)` "
                "or plain `@register` on an annotated function." % (cls,)
            )
        self.registry[cls] = func
        self._cache = {}
        return func

    def __call__(self, *args, **kw):
        if not args:
            raise TypeError("%s requires at least 1 positional argument" % self.__name__)
        return self.dispatch(type(args[0]))(*args, **kw)

    def __get__(self, instance, owner=None):
        return self


class singledispatchmethod:
    """Single-dispatch generic method descriptor."""

    def __init__(self, func):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError("%r is not callable or a descriptor" % func)
        self.dispatcher = singledispatch(func)
        self.func = func

    def register(self, cls, method=None):
        return self.dispatcher.register(cls, method)

    def __get__(self, obj, cls=None):
        dispatcher = self.dispatcher
        func = self.func

        def method(*args, **kwargs):
            implementation = dispatcher.dispatch(type(args[0]))
            get = getattr(implementation, "__get__", None)
            if get is not None:
                return get(obj, cls)(*args, **kwargs)
            return implementation(obj, *args, **kwargs)

        method.__isabstractmethod__ = self.__isabstractmethod__
        method.register = self.register
        update_wrapper(method, func)
        return method

    @property
    def __isabstractmethod__(self):
        return bool(getattr(self.func, "__isabstractmethod__", False))


_NOT_FOUND = object()


class cached_property:
    """A property computed once per instance and stored in the instance's ``__dict__``."""

    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                "(%r and %r)." % (self.attrname, name)
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except AttributeError:
            raise TypeError(
                "No '__dict__' attribute on %r instance to cache %r property."
                % (type(instance).__name__, self.attrname)
            ) from None
        value = cache.get(self.attrname, _NOT_FOUND)
        if value is _NOT_FOUND:
            value = self.func(instance)
            cache[self.attrname] = value
        return value


def recursive_repr(fillvalue="..."):
    """Decorate a ``__repr__`` so a self-referential structure prints ``fillvalue``."""

    def decorating_function(user_function):
        running = set()

        def wrapper(self):
            key = id(self)
            if key in running:
                return fillvalue
            running.add(key)
            try:
                return user_function(self)
            finally:
                running.discard(key)

        update_wrapper(wrapper, user_function)
        return wrapper

    return decorating_function
