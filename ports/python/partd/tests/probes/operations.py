"""Store small byte partitions in memory."""

import partd

store = partd.Dict()
store.append({"a": b"one"})
store.append({"a": b"two", "b": b"three"})
assert store.get(["a", "b"]) == [b"onetwo", b"three"]
store.delete(["a"])
assert store.get(["a"]) == [b""]
files = partd.File("/work/partitions")
files.append({"a": b"first", "b": b"second"})
files.append({"a": b"-more"})
assert files.get(["a", "b"]) == [b"first-more", b"second"]
files.delete(["a"])
assert files.get(["a"]) == [b""]
files.drop()
print("partd: memory and locked virtual-file append, retrieval and deletion passed")
