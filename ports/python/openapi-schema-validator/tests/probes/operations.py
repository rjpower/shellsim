"""Exercise local schema semantics and invalid input."""

from jsonschema import ValidationError
from openapi_schema_validator import OAS30Validator, OAS31Validator, validate

validate(None, {"type": "string", "nullable": True}, cls=OAS30Validator)
validate(7, {"type": ["integer", "null"]}, cls=OAS31Validator)
try:
    validate("seven", {"type": "integer"}, cls=OAS31Validator)
except ValidationError:
    pass
else:
    raise AssertionError("invalid OpenAPI value passed")
print("openapi-schema-validator: schema operations passed")
