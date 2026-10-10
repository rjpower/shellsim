"""Exercise guest allocation beyond the old image ceiling, including release."""

import ctypes

libc = ctypes.CDLL(None)
libc.malloc.argtypes = (ctypes.c_size_t,)
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = (ctypes.c_void_p,)
libc.free.restype = None
size = 80 * 1024**2
for _ in range(2):
    pointer = libc.malloc(size)
    assert pointer is not None
    try:
        allocation = (ctypes.c_ubyte * size).from_address(pointer)
        for offset in range(0, size, 65536):
            allocation[offset] = 123
        allocation[size - 1] = 124
        assert all(allocation[offset] == 123 for offset in range(0, size, 65536))
        assert allocation[size - 1] == 124
    finally:
        libc.free(pointer)
print("threaded CPython: allocation beyond 64 MiB passed")
