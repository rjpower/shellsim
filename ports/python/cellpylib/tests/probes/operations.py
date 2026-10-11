"""Evolve a fixed automaton and render it through the declared Agg backend."""

import io

import matplotlib

matplotlib.use("Agg")
import cellpylib as cpl
import matplotlib.pyplot as plt
import numpy as np

initial = cpl.init_simple(9)
result = cpl.evolve(initial, timesteps=3, apply_rule=lambda n, c, t: cpl.nks_rule(n, 30))
assert result.shape == (3, 9)
assert result[0].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0]
assert result[1].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0]
assert np.isin(result, [0, 1]).all()
cpl.plot(result, title="Rule 30", show=False)
buffer = io.BytesIO()
plt.savefig(buffer, format="png")
plt.close("all")
assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
try:
    cpl.init_random(5, n_randomized=6)
except ValueError:
    pass
else:
    raise AssertionError("oversized random initialization accepted")
print("cellpylib: rule 30 evolution, Agg plot and invalid size passed")
