"""Check fixed arbitrary precision arithmetic and integration in the guest."""

from mpmath import mp

with mp.workdps(40):
    assert abs(mp.sqrt(2) ** 2 - 2) < mp.mpf("1e-35")
    assert abs(mp.quad(lambda x: x * x, [0, 1]) - mp.mpf(1) / 3) < mp.mpf("1e-35")
assert mp.dps == 15
print("mpmath: precision context and numerical integration passed")
