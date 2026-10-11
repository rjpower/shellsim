"""Exercise a bounded public API in the actual guest."""

import jmespath

assert jmespath.search(
    "items[?score > `1`].name", {"items": [{"name": "a", "score": 1}, {"name": "b", "score": 2}]}
) == ["b"]
print("jmespath: guest public API probe passed")
