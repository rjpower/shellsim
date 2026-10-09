import ctypes
import subprocess
import zlib

api = ctypes.pythonapi.PyLong_FromLong
api.argtypes = [ctypes.c_long]
api.restype = ctypes.py_object
assert api(42) == 42
callback = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(lambda value: value + 2.0)
assert callback(1.5) == 3.5
assert zlib.decompress(zlib.compress(b"ready")) == b"ready"
child = subprocess.run(["python", "-c", "print(17)"], capture_output=True, text=True, check=True)
assert child.stdout == "17\n"
print("public ctypes zlib subprocess: ok")
