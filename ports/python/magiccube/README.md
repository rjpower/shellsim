# Magiccube 0.3.0

`uv run --no-project python -m ports python/magiccube --store target/ports-store --output target/magiccube-release`
selects the canonical graph. The shared pure-wheel helper verifies the recipe's
source hash, package metadata, pure tags and dependency metadata, then copies the
upstream wheel unchanged. Add that wheel to the ordinary curated package catalog.
NumPy is resolved by the public package installer from the selected catalog.

`verify.py --bundle <bundle> --universe <catalog> --uv <patched-uv>` installs
Magiccube through `CPythonRuntime.install_pypi` and runs the checked-in cube move
probe inside the guest. The port-local guest test uses the same public API.
It checks NumPy reductions and object arrays, cube moves and inverse restoration.
