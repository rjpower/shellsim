"""Serialize a closure without host execution or process workers."""

import cloudpickle


def factory(offset):
    return lambda value: value + offset


restored = cloudpickle.loads(cloudpickle.dumps(factory(7)))
assert restored(5) == 12
assert cloudpickle.loads(cloudpickle.dumps({"values": [1, 2]})) == {"values": [1, 2]}
print("cloudpickle: closure and ordinary data roundtrip passed")
