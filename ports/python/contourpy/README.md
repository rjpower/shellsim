# ContourPy

ContourPy 1.3.3 builds its upstream C++ extension against the target CPython
runtime. Pybind headers trigger Clang's unused-template warning; that specific
warning remains visible without making the upstream warnings-as-errors setting
reject those headers. Other compiler diagnostics keep their usual severity.
The threaded guest loader resolves preemptible calls to the extension's own
inline Pybind definitions when no earlier provider supplies them.

The guest probe covers serial contour extraction and invalid grid rejection.
Threaded contour execution is not qualified by this probe.
