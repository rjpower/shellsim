"""Call one real ctypes callback from the main Store and two pthread Stores."""

import ctypes
import gc
import threading

library = ctypes.CDLL("/lib/ctypes_threads_probe.so", use_errno=True)
callback_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int, ctypes.c_double, use_errno=True)
library.set_bias.argtypes = [ctypes.c_int]
library.set_bias.restype = None
library.get_bias.argtypes = []
library.get_bias.restype = ctypes.c_int
library.add.argtypes = [ctypes.c_int, ctypes.c_double]
library.add.restype = ctypes.c_double
library.call_callback.argtypes = [callback_type, ctypes.c_int, ctypes.c_double]
library.call_callback.restype = ctypes.c_double
library.call_from_pthread.argtypes = [callback_type, ctypes.c_int, ctypes.c_double]
library.call_from_pthread.restype = ctypes.c_double
observed = []
local = threading.local()


def callback(integer, real):
    assert ctypes.get_errno() == 37
    assert library.get_bias() == local.bias
    value = library.add(integer, real)
    ctypes.set_errno(73)
    observed.append((threading.get_ident(), local.bias))
    return value


closure = callback_type(callback)
old_pointer = ctypes.cast(closure, ctypes.c_void_p).value


def work(bias, function):
    local.bias = bias
    library.set_bias(bias)
    for _ in range(3):
        assert library.call_callback(function, 3, 2.5) == bias + 5.5
        assert ctypes.get_errno() == 73
    assert library.get_bias() == bias


work(10, closure)
barrier = threading.Barrier(3)


def threaded_work(bias, function):
    barrier.wait()
    work(bias, function)


threads = [threading.Thread(target=threaded_work, args=(value, closure)) for value in (100, 200)]
for thread in threads:
    thread.start()
barrier.wait()
for thread in threads:
    thread.join()
assert sorted(bias for _, bias in observed) == [10] * 3 + [100] * 3 + [200] * 3
assert len({identity for identity, _ in observed}) == 3
assert library.get_bias() == 10
foreign_calls = []


def foreign_callback(integer, real):
    assert ctypes.get_errno() == 37
    assert library.get_bias() == 300
    value = library.add(integer, real)
    ctypes.set_errno(73)
    foreign_calls.append(threading.get_ident())
    return value


foreign_closure = callback_type(foreign_callback)
assert library.call_from_pthread(foreign_closure, 3, 2.5) == 305.5
assert len(foreign_calls) == 1
assert foreign_calls[0] != threading.get_ident()
assert library.get_bias() == 10
api = ctypes.pythonapi.PyLong_FromLong
api.argtypes = [ctypes.c_long]
api.restype = ctypes.py_object
assert api(42) == 42
del closure
gc.collect()
replacement = callback_type(lambda integer, real: integer + real)
assert ctypes.cast(replacement, ctypes.c_void_p).value != old_pointer
assert replacement(3, 2.5) == 5.5
print("threaded ctypes: callback, nested CDLL call, TLS, errno passed")
