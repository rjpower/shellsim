"""Exercise local schema semantics and invalid input."""

from jsonschema import Draft202012Validator, ValidationError, validate
from referencing import Registry, Resource

resource = Resource.from_contents(
    {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "urn:count", "type": "integer", "minimum": 0}
)
validator = Draft202012Validator({"$ref": "urn:count"}, registry=Registry().with_resource("urn:count", resource))
assert validator.is_valid(4) and not validator.is_valid(-1)
schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "additionalProperties": False}
validate({"n": 7}, schema)
for value in [{}, {"n": "7"}, {"n": 7, "extra": True}]:
    try:
        validate(value, schema)
    except ValidationError:
        pass
    else:
        raise AssertionError("invalid value passed validation")
print("jsonschema: schema operations passed")
