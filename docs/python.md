# Python in shellsim

Shellsim implements Python source directly in its deterministic, resource-bounded environment.
The goal is to run ordinary Python used in agent tasks without granting access to host Python,
native extensions, files, processes, networking, environment variables, locale, or time.

The language runtime is intended to be mostly complete. Library compatibility follows a harder
boundary: a module is included when shellsim can provide a coherent, useful implementation of its
ordinary behavior. Missing modules and APIs fail at import or attribute lookup instead of exposing
plausible but inconsistent stubs.

## Run Python

The `python`, `python3`, and `python3.14` commands run source from `-c`, stdin, or a VFS file:

```sh
shellsim -c 'python3 -c "print(sum(x*x for x in range(5)))"'
shellsim --root ./project -c 'python3.14 /work/main.py'
shellsim-python ./project/main.py -- arg1
shellsim-python ./project/tests --pytest
```

`shellsim-python` imports the selected trusted project into `/work` before execution. It supports
`--entry`, `--pytest`, `--root`, resource limits, and `--json`. Simulated imports remain confined
to frozen modules and the virtual filesystem.

The runtime supports functions and closures, classes and descriptors, exceptions and context
managers, comprehensions and generators, arbitrary-precision integers, mutable containers,
f-strings, VFS imports, and the common language protocols needed by real scripts. `pytest` and
`unittest` provide bounded runners for straightforward test files. Unsupported syntax and runner
features produce an error rather than a false passing result.

## Runtime model

```text
source -> UTF-8 lexer -> AST parser -> bytecode compiler -> metered stack VM
                                                        -> native modules
                                                        -> modeled capabilities
```

The bytecode is shellsim's internal semantic format, not CPython bytecode. Values are immediate
scalars or typed objects in an interpreter-owned arena. User-visible identity, mutation, type
lookup, descriptors, and operator slots are modeled explicitly. CPU is charged per instruction
and native loop; source, recursion, calls, allocation, iteration, and output are bounded.

Native modules exchange a type-erased `PyValue` and recover checked views such as `PyNumber`,
`PyString`, `PyList`, or `PyArray`. They receive only the capabilities declared by `PyRuntime`.
A pure algorithm cannot acquire VFS, process, clock, or network access accidentally.

## Adding a module

Use a frozen Python module when the behavior composes naturally from supported Python. Use a Rust
native module for compact algorithms, interpreter-owned objects, or an explicit modeled
capability. Native definitions use declarative function, method, type, and value tables.

For either form:

1. define the useful supported surface and the explicit unsupported frontier;
2. validate arguments and reject unknown keywords;
3. meter loops and reserve result storage before allocation;
4. keep mutable interpreter layouts behind checked runtime views;
5. add differential tests against the matching CPython behavior where deterministic.

Do not add an importable placeholder for a module whose central contract is absent. For example,
an empty `sqlite3` namespace is less useful than a clear import failure because callers otherwise
cannot tell which database semantics are real.

The minimal NumPy design follows these rules in [numpy.md](numpy.md).
