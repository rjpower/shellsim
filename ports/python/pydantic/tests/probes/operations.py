"""Exercise real model/ASGI behavior without host network access."""

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class Item(BaseModel):
    n: int
    name: str = Field(min_length=2)


item = Item.model_validate({"n": "7", "name": "ok"})
assert item.n == 7 and item.model_dump() == {"n": 7, "name": "ok"}
assert Item.model_validate_json(item.model_dump_json()) == item
assert TypeAdapter(list[int]).validate_python(["1", 2]) == [1, 2]
assert Item.model_json_schema()["properties"]["n"]["type"] == "integer"
try:
    Item(n="seven", name="x")
except ValidationError as error:
    assert len(error.errors()) == 2
else:
    raise AssertionError("invalid model accepted")

print("pydantic: operations passed")
