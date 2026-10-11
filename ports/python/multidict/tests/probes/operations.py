"""Exercise the declared local guest API."""

from multidict import CIMultiDict, MultiDict, MultiDictProxy, istr
from multidict._multidict import MultiDict as NativeDict

assert MultiDict is NativeDict
values = MultiDict([("x", "1"), ("x", "2")])
assert values.getall("x") == ["1", "2"]
values.add("x", "3")
assert list(values.items()) == [("x", "1"), ("x", "2"), ("x", "3")]
assert CIMultiDict({"Content-Type": "text/plain"})["content-type"] == "text/plain"
assert list(values.keys()) == ["x"]
assert list(values.values()) == ["1", "2", "3"]
assert values.copy().getall("x") == ["1", "2", "3"]
proxy = MultiDictProxy(values)
assert list(proxy.keys()) == list(values.keys())
assert list(proxy.items()) == list(values.items())
assert list(proxy.values()) == list(values.values())
assert proxy.copy().getall("x") == ["1", "2", "3"]
assert values.to_dict() == proxy.to_dict() == {"x": ["1", "2", "3"]}
for item in (values, istr("key")):
    constructor, arguments = item.__reduce__()
    restored = constructor(*arguments)
    assert restored == item
try:
    proxy.__reduce__()
except TypeError:
    pass
else:
    raise AssertionError("unpicklable proxy was accepted")
assert values.__sizeof__() > 0
copy = values.copy()
assert copy.popitem() in [("x", "1"), ("x", "2"), ("x", "3")]
copy.clear()
assert len(copy) == 0
try:
    values["missing"]
except KeyError:
    pass
else:
    raise AssertionError("missing key returned")
print("multidict: local guest operations passed")
