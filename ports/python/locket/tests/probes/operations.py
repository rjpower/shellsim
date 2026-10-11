"""Acquire and release one lock against the guest virtual filesystem."""

import locket

with locket.lock_file("/tmp/port-lock", timeout=1):
    pass
with locket.lock_file("/tmp/port-lock"):
    pass
print("locket: guest file lock passed")
