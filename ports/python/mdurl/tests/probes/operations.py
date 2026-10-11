"""Parse URLs and round-trip escaped text."""

import mdurl

url = mdurl.parse("https://example.org/path?q=1#part")
assert url.hostname == "example.org" and url.pathname == "/path"
assert url.search == "?q=1" and url.hash == "#part"
assert mdurl.decode(mdurl.encode("a b")) == "a b"
print("mdurl: URL parsing and escaping passed")
