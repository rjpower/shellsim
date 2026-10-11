# Pandas

Build Pandas 2.3.3 with the admitted target NumPy headers and pinned source
version. Tag its Python package directories for runtime installation.
Qualify grouping, joins, CSV through virtual files and timezone conversion.
WASI has no mmap extension; ordinary IO remains available and an explicit
`memory_map=True` request raises `NotImplementedError`. Other platforms retain
upstream mmap imports. Optional accelerators and storage formats are unqualified.
