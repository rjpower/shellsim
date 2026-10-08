"""Shallow and deep copies of arbitrary objects.

Builtin containers copy structurally; instances copy through ``__copy__``/``__deepcopy__`` when
defined, otherwise by duplicating their attribute dictionary. Deep copies share a memo keyed by
object identity, so cycles and shared references are preserved.
"""

__all__ = ["Error", "copy", "deepcopy", "replace"]

from copyreg import dispatch_table


class Error(Exception):
    pass


error = Error

_ATOMIC = (type(None), int, float, bool, complex, str, bytes, type, range, slice, frozenset,
           type(Ellipsis), type(NotImplemented))


def copy(x):
    cls = type(x)
    if cls in _ATOMIC or isinstance(x, _ATOMIC):
        return x
    if cls is list:
        return list(x)
    if cls is dict:
        return dict(x)
    if cls is set:
        return set(x)
    if cls is tuple:
        return x
    if cls is bytearray:
        return bytearray(x)
    copier = getattr(x, "__copy__", None)
    if copier is not None:
        return copier()
    reductor = dispatch_table.get(cls)
    if reductor is not None:
        return _reconstruct(x, None, *reductor(x))
    if isinstance(x, list):
        result = cls(x)
        _copy_attributes(x, result)
        return result
    if isinstance(x, dict):
        result = cls(x)
        _copy_attributes(x, result)
        return result
    if isinstance(x, set):
        return cls(x)
    if isinstance(x, tuple):
        return cls(x)
    return _copy_instance(x)


def _copy_attributes(source, target):
    state = getattr(source, "__dict__", None)
    if state:
        for key, value in state.items():
            setattr(target, key, value)


def _new_instance(x):
    cls = type(x)
    constructor = getattr(x, "__getnewargs__", None)
    if constructor is not None:
        return cls.__new__(cls, *constructor())
    return cls.__new__(cls)


def _copy_instance(x):
    result = _new_instance(x)
    state = getattr(x, "__dict__", None)
    setstate = getattr(result, "__setstate__", None)
    if setstate is not None and state is not None:
        setstate(dict(state))
        return result
    if state is not None:
        for key, value in state.items():
            setattr(result, key, value)
    for name in getattr(type(x), "__slots__", ()):
        if hasattr(x, name):
            setattr(result, name, getattr(x, name))
    return result


def deepcopy(x, memo=None, _nil=[]):
    if memo is None:
        memo = {}
    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y
    cls = type(x)
    if cls in _ATOMIC or isinstance(x, _ATOMIC):
        return x
    if cls is list or cls is dict or cls is set or cls is tuple or cls is bytearray:
        y = _deepcopy_builtin(x, memo)
    else:
        copier = getattr(x, "__deepcopy__", None)
        if copier is not None:
            y = copier(memo)
        else:
            reductor = dispatch_table.get(cls)
            if reductor is not None:
                y = _reconstruct(x, memo, *reductor(x))
            else:
                y = _deepcopy_instance(x, memo)
    if y is not x:
        memo[d] = y
        _keep_alive(x, memo)
    return y


def _deepcopy_builtin(x, memo):
    cls = type(x)
    if cls is list:
        y = []
        memo[id(x)] = y
        for item in x:
            y.append(deepcopy(item, memo))
        return y
    if cls is dict:
        y = {}
        memo[id(x)] = y
        for key, value in x.items():
            y[deepcopy(key, memo)] = deepcopy(value, memo)
        return y
    if cls is set:
        y = set()
        memo[id(x)] = y
        for item in x:
            y.add(deepcopy(item, memo))
        return y
    if cls is bytearray:
        return bytearray(x)
    items = [deepcopy(item, memo) for item in x]
    for original, copied in zip(x, items):
        if original is not copied:
            return tuple(items)
    return x


def _deepcopy_instance(x, memo):
    result = _new_instance(x)
    memo[id(x)] = result
    if isinstance(x, list):
        for item in x:
            result.append(deepcopy(item, memo))
    elif isinstance(x, dict):
        for key, value in x.items():
            result[deepcopy(key, memo)] = deepcopy(value, memo)
    elif isinstance(x, set):
        for item in x:
            result.add(deepcopy(item, memo))
    state = getattr(x, "__dict__", None)
    if state is not None:
        copied = {key: deepcopy(value, memo) for key, value in state.items()}
        setstate = getattr(result, "__setstate__", None)
        if setstate is not None:
            setstate(copied)
        else:
            for key, value in copied.items():
                setattr(result, key, value)
    for name in getattr(type(x), "__slots__", ()):
        if hasattr(x, name):
            setattr(result, name, deepcopy(getattr(x, name), memo))
    return result


def _reconstruct(x, memo, func, args, state=None, listiter=None, dictiter=None):
    deep = memo is not None
    if deep and args:
        args = deepcopy(args, memo)
    y = func(*args)
    if deep:
        memo[id(x)] = y
    if state is not None:
        if deep:
            state = deepcopy(state, memo)
        setstate = getattr(y, "__setstate__", None)
        if setstate is not None:
            setstate(state)
        else:
            for key, value in state.items():
                setattr(y, key, value)
    if listiter is not None:
        for item in listiter:
            y.append(deepcopy(item, memo) if deep else item)
    if dictiter is not None:
        for key, value in dictiter:
            if deep:
                key, value = deepcopy(key, memo), deepcopy(value, memo)
            y[key] = value
    return y


def _keep_alive(x, memo):
    try:
        memo[id(memo)].append(x)
    except KeyError:
        memo[id(memo)] = [x]


def replace(obj, /, **changes):
    """``obj.__replace__(**changes)``: a copy with the named fields replaced."""
    cls = type(obj)
    func = getattr(cls, "__replace__", None)
    if func is None:
        raise TypeError("replace() does not support %s objects" % cls.__name__)
    return func(obj, **changes)
