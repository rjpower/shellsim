"""Exercise a bounded public API in the actual guest."""

import xmltodict

data = xmltodict.parse('<root><value key="a">text</value></root>')
assert data["root"]["value"] == {"@key": "a", "#text": "text"}
assert xmltodict.parse(xmltodict.unparse(data))["root"] == data["root"]
print("xmltodict: guest public API probe passed")
