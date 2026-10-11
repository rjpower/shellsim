"""Exercise a bounded public API in the actual guest."""

from importlib_metadata import metadata, version

assert version("importlib-metadata") == "9.0.1"
assert metadata("importlib-metadata")["Name"] == "importlib_metadata"
print("importlib-metadata: guest public API probe passed")
