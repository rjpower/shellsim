"""Exercise local schema semantics and invalid input."""

from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource
from referencing.jsonschema import DRAFT202012

resource = Resource.from_contents(
    {"$id": "urn:example", "$defs": {"answer": {"type": "integer"}}}, default_specification=DRAFT202012
)
empty = Registry()
registry = empty.with_resource("urn:example", resource)
assert registry.resolver().lookup("urn:example#/$defs/answer").contents == {"type": "integer"}
assert len(empty) == 0 and len(registry) == 1
try:
    empty.contents("urn:example")
except NoSuchResource:
    pass
else:
    raise AssertionError("empty registry found a resource")
print("referencing: schema operations passed")
