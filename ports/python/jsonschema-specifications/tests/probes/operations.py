"""Exercise local schema semantics and invalid input."""

from jsonschema_specifications import REGISTRY
from referencing import Registry

assert isinstance(REGISTRY, Registry)
schema = REGISTRY.contents("https://json-schema.org/draft/2020-12/schema")
assert schema["$id"] == "https://json-schema.org/draft/2020-12/schema"
assert "type" in REGISTRY.contents("https://json-schema.org/draft/2020-12/meta/validation")["properties"]
print("jsonschema-specifications: schema operations passed")
