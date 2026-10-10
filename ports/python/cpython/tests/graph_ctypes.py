"""Exercise scalar foreign calls and callbacks through the published FFI closure."""

import ctypes

libc = ctypes.CDLL(None)
libc.abs.argtypes = (ctypes.c_int,)
libc.abs.restype = ctypes.c_int
assert libc.abs(-17) == 17
callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(lambda value: value + 3)
assert callback(4) == 7
print("ctypes scalar calls and callbacks passed")
