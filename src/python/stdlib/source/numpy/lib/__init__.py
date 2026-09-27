"""`numpy.lib`: the `.npy`/`.npz` file format (`format`) and its readers/writers (`npyio`).

NumPy's `numpy.lib` is a larger grab-bag of submodules; shellsim exposes only the file-I/O
pieces its top-level `numpy` package re-exports (`save`, `load`, `savetxt`, ...), plus the
two submodules programs import directly.
"""

from numpy.lib import format
from numpy.lib import npyio

__all__ = ["format", "npyio"]
