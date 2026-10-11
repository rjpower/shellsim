"""Exercise a bounded public API inside the actual guest."""

from urllib3.util import parse_url
from urllib3.util.retry import Retry

assert parse_url("https://example.invalid/a?b=1").host == "example.invalid"
assert Retry(total=2).increment(method="GET").total == 1
print("urllib3: public API probe passed")
