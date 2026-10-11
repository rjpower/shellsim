"""Exercise a bounded public API in the actual guest."""

from blinker import Signal

event = Signal()


def receive(sender, value):
    return value + 1


event.connect(receive)
assert event.send("probe", value=3) == [(receive, 4)]
print("blinker: guest public API probe passed")
