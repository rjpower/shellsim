import importlib.metadata

import packaging
from packaging.version import Version

assert importlib.metadata.version("packaging") == "26.3"
assert packaging.__version__ == "26.3"
assert Version("1.0rc1") < Version("1.0")
