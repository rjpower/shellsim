"""Exercise the real SDK callback registry without claiming a guest fork API."""

import ctypes
import subprocess
import sys

libc = ctypes.CDLL(None)
callback_type = ctypes.CFUNCTYPE(None)
register = libc.pthread_atfork
register.argtypes = (callback_type, callback_type, callback_type)
register.restype = ctypes.c_int
handlers = libc.__fork_handler
handlers.argtypes = (ctypes.c_int,)
handlers.restype = None

events = []
callbacks = []
for number in (1, 2):
    group = tuple(
        callback_type(lambda label=f"{phase}{number}": events.append(label)) for phase in ("prepare", "parent", "child")
    )
    callbacks.extend(group)
    assert register(*group) == 0
assert register(callback_type(), callback_type(), callback_type()) == 0

# Spawn creates a fresh executable, so fork callbacks must not run in the parent.
result = subprocess.run([sys.executable, "-c", "print(11)"], capture_output=True, check=True)
assert result.stdout == b"11\n" and result.stderr == b""
assert events == []

# Exercise musl's dispatcher explicitly: this tests the registry and ordering,
# without simulating a fork or pretending that spawn invokes these phases.
handlers(-1)
assert events == ["prepare2", "prepare1"]
handlers(0)
assert events == ["prepare2", "prepare1", "parent1", "parent2"]
events.clear()
handlers(-1)
handlers(1)
assert events == ["prepare2", "prepare1", "child1", "child2"]
print("threaded CPython: real atfork registry ordering and spawn separation passed")
