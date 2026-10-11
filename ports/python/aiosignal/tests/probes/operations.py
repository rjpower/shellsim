"""Exercise the declared local guest API."""

from aiosignal import Signal

seen = []
signal = Signal("owner")


async def callback(value):
    seen.append(value)


signal.append(callback)
signal.freeze()
operation = signal.send(7)
try:
    operation.send(None)
except StopIteration:
    pass
else:
    raise AssertionError("immediate callback yielded")
assert seen == [7]
try:
    signal.append(callback)
except RuntimeError:
    pass
else:
    raise AssertionError("frozen signal was mutable")
print("aiosignal: local guest operations passed")
