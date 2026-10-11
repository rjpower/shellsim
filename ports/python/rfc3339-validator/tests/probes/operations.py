"""Exercise local schema semantics and invalid input."""

from rfc3339_validator import validate_rfc3339

assert validate_rfc3339("2026-10-11T06:00:00Z")
assert not validate_rfc3339("2026-13-40T25:00:00Z")
assert not validate_rfc3339("yesterday")
print("rfc3339-validator: schema operations passed")
