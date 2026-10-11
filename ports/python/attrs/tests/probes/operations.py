"""Exercise a bounded public API inside the actual guest."""

from attrs import define, field, validators


@define
class Record:
    count: int = field(validator=validators.instance_of(int))


assert Record(3).count == 3
try:
    Record("wrong")
except TypeError:
    pass
else:
    raise AssertionError("invalid field accepted")
print("attrs: public API probe passed")
