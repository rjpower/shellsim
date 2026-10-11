"""Exercise a bounded public API in the actual guest."""

from absl import flags

values = flags.FlagValues()
flags.DEFINE_integer("count", 1, "A bounded count", flag_values=values)
values(["probe", "--count=3"])
assert values.count == 3
try:
    values(["probe", "--count=wrong"])
except flags.IllegalFlagValueError:
    pass
else:
    raise AssertionError("invalid flag accepted")
print("absl-py: guest public API probe passed")
