"""Exercise a bounded public API in the actual guest."""

from marshmallow import Schema, ValidationError, fields


class Record(Schema):
    count = fields.Integer(required=True)


assert Record().load({"count": "3"}) == {"count": 3}
try:
    Record().load({"count": "bad"})
except ValidationError:
    pass
else:
    raise AssertionError("invalid schema input accepted")
print("marshmallow: guest public API probe passed")
