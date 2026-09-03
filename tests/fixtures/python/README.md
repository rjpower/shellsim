# Python corpus fixtures

These reduced fixtures preserve semantic feature combinations observed in public TaskTrove/TBLite
tasks without embedding task answers or teaching the runtime their expected output. Each fixture
must also execute under CPython 3.14; shellsim tests compare observable status/stdout/stderr.

`task_ordering_bootstrap.py` is the first reduction from `build-system-task-ordering`. It covers the
input-scanning portion: functions, indented branching and loops, short-circuiting, string/list
methods, indexing, and stable sorting. Later reductions will add dictionary/set mutation, nested
functions and closures, classes, comprehensions, JSON, and VFS import loading.
