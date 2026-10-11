"""Exercise a bounded public API inside the actual guest."""

from requests import Request

request = Request("GET", "https://example.invalid/path", params={"q": "a b"}).prepare()
assert request.url == "https://example.invalid/path?q=a+b"
assert request.method == "GET"
print("requests: public API probe passed")
