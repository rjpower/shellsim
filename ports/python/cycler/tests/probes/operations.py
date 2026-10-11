"""Exercise deterministic cycle composition and malformed input rejection."""

from cycler import cycler

cycle = cycler(color=["red", "blue"]) + cycler(linewidth=[1, 2])
assert list(cycle) == [{"color": "red", "linewidth": 1}, {"color": "blue", "linewidth": 2}]
assert len(cycler(color=["red", "blue"]) * cycler(marker=["o", "x"])) == 4
try:
    cycler(color=["red"]) + cycler(marker=["o", "x"])
except ValueError:
    pass
else:
    raise AssertionError("unequal cycles accepted")
print("cycler: composition and invalid lengths passed")
