"""Exercise CFFI allocation, scalar calls and callbacks in the guest."""

from cffi import FFI

ffi = FFI()
values = ffi.new("int[]", [1, 2, 3])
assert sum(values) == 6
pointer = ffi.new("int *", 7)
assert pointer[0] == 7
pointer[0] = 9
assert pointer[0] == 9
text = ffi.new("char[]", b"virtual")
assert ffi.string(text) == b"virtual"
ffi.cdef("const char *Py_GetVersion(void);")
assert ffi.string(ffi.dlopen(None).Py_GetVersion()).startswith(b"3.13.")
callback = ffi.callback("int(int)", lambda value: value + 1)
assert callback(7) == 8
print("cffi: guest allocation, scalar call and callback passed")
