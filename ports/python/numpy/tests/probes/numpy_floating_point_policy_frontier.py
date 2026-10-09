import numpy as np

np.seterr(all="raise")
# This pinned experimental profile cannot detect the hardware status flags.
assert np.isinf(np.divide(1.0, 0.0))
print("floating-point policy gap reproduced")
