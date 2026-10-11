"""Exercise real model/ASGI behavior without host network access."""

from pydantic_core import SchemaSerializer, SchemaValidator, ValidationError, core_schema, from_json

schema = core_schema.typed_dict_schema({"n": core_schema.typed_dict_field(core_schema.int_schema())})
validator = SchemaValidator(schema)
assert validator.validate_python({"n": "7"}) == {"n": 7}
assert validator.validate_json('{"n": 8}') == {"n": 8}
assert SchemaSerializer(schema).to_json({"n": 9}) == b'{"n":9}'
try:
    validator.validate_python({"n": "seven"})
except ValidationError as error:
    assert error.errors()[0]["type"] == "int_parsing"
else:
    raise AssertionError("invalid integer accepted")
try:
    from_json(b"{bad")
except ValueError:
    pass
else:
    raise AssertionError("invalid JSON accepted")

print("pydantic-core: operations passed")
