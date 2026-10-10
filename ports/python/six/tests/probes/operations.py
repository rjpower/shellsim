import six

assert six.PY3
assert list(six.iteritems({"answer": 42})) == [("answer", 42)]
print("Six iteration passed")
