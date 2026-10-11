"""Exercise lazy composition and bounded grouping."""

from toolz import compose, frequencies, groupby, partition_all

assert compose(lambda value: value + 1, lambda value: value * 2)(3) == 7
assert frequencies("ababa") == {"a": 3, "b": 2}
assert groupby(lambda value: value % 2, [1, 2, 3]) == {1: [1, 3], 0: [2]}
assert list(partition_all(2, [1, 2, 3])) == [(1, 2), (3,)]
print("toolz: composition, frequencies, grouping and partitions passed")
