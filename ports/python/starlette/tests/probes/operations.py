"""Exercise local public behavior in the actual guest."""

from starlette.datastructures import URL, Headers, QueryParams
from starlette.responses import JSONResponse
from starlette.routing import compile_path

assert QueryParams("x=1&x=2").getlist("x") == ["1", "2"]
assert Headers({"X-Count": "7"})["x-count"] == "7"
assert URL("http://example.test/item?x=1").path == "/item"
assert JSONResponse({"value": 7}).body == b'{"value":7}'
expression, _, converters = compile_path("/item/{value:int}")
assert converters["value"].convert(expression.match("/item/7").group("value")) == 7
print("starlette: guest probe passed")
