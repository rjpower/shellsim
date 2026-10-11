"""Exercise a bounded public API in the actual guest."""

import tomli_w
import tomllib

data = {"task": {"count": 3, "names": ["a", "b"]}}
assert tomllib.loads(tomli_w.dumps(data)) == data
print("tomli-w: guest public API probe passed")
