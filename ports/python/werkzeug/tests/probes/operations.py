"""Exercise local public behavior in the actual guest."""

from werkzeug.routing import Map, Rule
from werkzeug.test import Client
from werkzeug.wrappers import Response

mapping = Map([Rule("/item/<int:value>", endpoint="item")])
assert mapping.bind("example.test").match("/item/7") == ("item", {"value": 7})
client = Client(lambda environ, start: Response("virtual", status=201)(environ, start), Response)
assert client.get("/").status_code == 201
assert client.get("/").get_data(as_text=True) == "virtual"
print("werkzeug: guest probe passed")
