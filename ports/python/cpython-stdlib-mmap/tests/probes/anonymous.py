"""Exercise anonymous buffers, read-only file snapshots and unavailable flush."""

import mmap

mapping = mmap.mmap(-1, 4096, flags=mmap.MAP_PRIVATE)
mapping[:4] = b"data"
view = memoryview(mapping)
view[4] = ord("!")
mapping.seek(0)
assert mapping.read(5) == b"data!"
try:
    mapping.close()
except BufferError:
    pass
else:
    raise AssertionError("mapping closed while a view remained exported")
view.release()
mapping.close()
try:
    mapping.read(1)
except ValueError:
    pass
else:
    raise AssertionError("closed mapping remained readable")

with open("/work/mmap-data", "w+b") as file:
    file.write(b"unchanged")
    file.flush()
    snapshot = mmap.mmap(file.fileno(), 9, access=mmap.ACCESS_READ)
    assert snapshot[:] == b"unchanged"
    try:
        snapshot[0] = ord("x")
    except TypeError:
        pass
    else:
        raise AssertionError("read-only mapping accepted a write")
    snapshot.close()
    mapping = mmap.mmap(file.fileno(), 9)
    mapping[0] = ord("x")
    try:
        mapping.flush()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unavailable mapping flush succeeded")
    mapping.close()
    file.seek(0)
    assert file.read() == b"unchanged"
print("mmap anonymous buffers, lifetime and read-only snapshots passed")
