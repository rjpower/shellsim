"""Exercise persistent containers and invalid input through the Rust extension."""

from rpds import HashTrieMap, HashTrieSet, List

empty = HashTrieMap()
original = HashTrieMap({"x": 7, "y": 9})
updated = original.insert("x", 11).remove("y")
assert dict(original) == {"x": 7, "y": 9}
assert dict(updated) == {"x": 11} and len(empty) == 0
assert HashTrieSet([1, 2]).insert(3).difference(HashTrieSet([2])) == HashTrieSet([1, 3])
assert list(List([1, 2]).push_front(0)) == [0, 1, 2]
try:
    original["missing"]
except KeyError:
    pass
else:
    raise AssertionError("missing key returned")
print("rpds: persistent containers passed")
