"""Exercise a bounded public API in the actual guest."""

from rfc3339_validator import validate_rfc3339

assert validate_rfc3339("2026-01-02T03:04:05Z")
assert not validate_rfc3339("2026-13-02T03:04:05Z")
print("rfc3339-validator: guest public API probe passed")
