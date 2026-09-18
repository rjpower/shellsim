# Python compatibility corpora

This directory contains manifest-driven differential probes and reduced real-world programs. They
need Rust-side orchestration or checked output, so they are separate from the portable
`tests/python/test_*.py` assertion suites. Valid compatibility cases must also execute under
CPython 3.14; shellsim tests compare observable status, stdout, and stderr.

`task_ordering_bootstrap.py` is the first reduction from `build-system-task-ordering`. It covers the
input-scanning portion: functions, indented branching and loops, short-circuiting, string/list
methods, indexing, and stable sorting. Later reductions will add dictionary/set mutation, nested
functions and closures, classes, comprehensions, JSON, and VFS import loading.
