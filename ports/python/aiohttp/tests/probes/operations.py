"""Exercise the declared local guest API."""

from aiohttp import FormData, web
from aiohttp._http_parser import HttpRequestParser
from aiohttp._websocket.mask import _websocket_mask_cython
from aiohttp.http_exceptions import BadHttpMessage

response = web.json_response({"value": 7}, status=201)
assert response.status == 201 and response.content_type == "application/json"
assert response.body == b'{"value": 7}'
form = FormData({"x": "a b"})()
assert form.size == 5

value = bytearray(b"abcd")
_websocket_mask_cython(bytes([1, 2, 3, 4]), value)
assert value == bytearray([96, 96, 96, 96])

parser = HttpRequestParser(None, None, 65536)
messages, upgraded, tail = parser.feed_data(b"GET /item?q=7 HTTP/1.1\r\nHost: example.test\r\n\r\n")
assert not upgraded and tail == b""
message, payload = messages[0]
assert message.method == "GET" and message.path == "/item?q=7"
assert message.headers["Host"] == "example.test"
assert payload.at_eof()
try:
    HttpRequestParser(None, None, 65536).feed_data(b"INVALID / HTTP/1.1\r\n\r\n")
except BadHttpMessage:
    pass
else:
    raise AssertionError("invalid HTTP method accepted")
print("aiohttp: local guest operations passed")
