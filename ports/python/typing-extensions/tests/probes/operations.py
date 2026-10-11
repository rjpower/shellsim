"""Exercise a bounded public API inside the actual guest."""

from typing_extensions import TypedDict, get_type_hints


class Record(TypedDict):
    count: int


assert get_type_hints(Record) == {"count": int}
assert Record(count=3) == {"count": 3}
print("typing-extensions: public API probe passed")
