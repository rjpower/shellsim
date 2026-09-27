"""Placeholder for SciPy subpackages shellsim does not implement.

`frozen.rs` maps every unsupported `scipy` subpackage name to this module, so importing any of
them (`scipy.sparse`, `scipy.optimize`, ...) runs this file under that dotted name and fails
here, before any of its real functionality could be reached. Some supported code paths import an
unsupported subpackage on demand instead of failing at start-up: `rv_continuous.fit` needs
`scipy.optimize` and continuous `expect` needs `scipy.integrate`, so shellsim's versions of them
raise this same error only when a caller reaches the unsupported computation.
"""

raise NotImplementedError(f"{__name__} is not supported by shellsim's SciPy")
