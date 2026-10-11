"""Use an in-memory filesystem without host network or storage."""

import fsspec

filesystem = fsspec.filesystem("memory")
filesystem.pipe("/probe/data", b"virtual")
assert filesystem.cat("/probe/data") == b"virtual"
assert filesystem.find("/probe") == ["/probe/data"]
try:
    filesystem.cat("/missing")
except FileNotFoundError:
    pass
else:
    raise AssertionError("missing file accepted")
print("fsspec: memory filesystem and missing path passed")
