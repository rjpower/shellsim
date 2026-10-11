"""Exercise the declared local guest API."""

from yarl import URL
from yarl._quoting_c import _Quoter

assert _Quoter()("a b") == "a%20b"
url = URL("https://example.test/a%20b?x=1&x=2")
assert url.path == "/a b" and url.query.getall("x") == ["1", "2"]
assert str(url.with_query({"q": "a b"})) == "https://example.test/a%20b?q=a+b"
assert str(URL("https://example.test/") / "x y") == "https://example.test/x%20y"
print("yarl: local guest operations passed")
