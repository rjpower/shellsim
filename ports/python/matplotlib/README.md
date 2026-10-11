# Matplotlib

Matplotlib 3.10.8 uses the existing Python Meson adapter and the declared target
FreeType and reentrant Qhull libraries. The source-release patch sets the pinned
3.10.8 version directly, avoiding VCS discovery before Meson initializes its
Python bindings. Ordinary installed meson-python provides build metadata;
host binary hashes are not part of product cache keys.

The runtime defaults to Agg. The guest probe renders an 80×80 PNG with text
and exercises Qhull triangulation. The source patches use explicit signed
buffer dimensions on 32-bit WASI and omit the Tk display extension. GUI
backends and host display access are outside this port's qualification.
