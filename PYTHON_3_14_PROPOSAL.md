# Proposal: a small, safe Python 3.14 emulator for shellsim

Status: implementation proposal and measured status, 2026-09-02

## Implementation status

The implementation now has one source-to-execution path. In addition to the `python3.14` command,
VFS-script, and shebang entrypoints, the engine has source spans, a UTF-8/indentation-aware lexer,
owned AST, recursive-descent/Pratt parser, typed semantic bytecode, and a VM that charges CPU per
instruction. Unsupported input receives a front-end or VM diagnostic and is never handed to the
former text-splitting scalar evaluator.

The language/runtime surface currently includes scalar and container displays; arena-backed
lists, tuples, insertion-ordered dictionaries, and sets; centralized truth/repr/equality/ordering/
containment protocols; mutable item assignment and common string/list/dict/set methods; Python
short-circuit and chained comparison behavior; indented `if`/`elif`/`else`, `while`, `for`/`else`,
`break`, and `continue`; functions, recursion, returns, and persistent parent-linked lexical
scopes. It also supports recursive tuple/list unpacking, name and subscript `+=`/`-=`, erased basic
annotations, and the generic `range`, `enumerate`, `zip`, `list`, `tuple`, `set`, `bool`, `any`, and
`all` builtins. Positional and keyword calls use one VM protocol; nested functions can update the
nearest enclosing binding with `nonlocal`; basic user classes provide independent instances,
attributes, bound methods, recursive method calls, and `__init__`. Simple `.py` modules load only
from shellsim's VFS, retain isolated module globals, cache
per invocation, and support closures over module and function scopes. Computed string/container
growth reserves modeled memory before mutation, call/expression depth is bounded, and loops consume
fuel on every bytecode instruction.

The current language slice includes functions, closures, recursion, classes and bound methods,
comprehensions, suspended generators, exceptions, `try`/`except`/`else`/`finally`, context managers,
`assert`, decorators, definition-time defaults, `*args`, starred assignment/calls, f-strings, and
VFS-only imports. The runtime deliberately models only the semantics it can enforce: generator
expressions are eagerly materialized and generators cannot suspend across cleanup regions. The
object model provides C3 inheritance, cached protocol slots, descriptors, `super`, builtin
subclass layouts, and metaclass construction hooks within the explicitly supported syntax.

The requested-library probe gate is now 18/18 exact probes against CPython 3.14 when available.
Those probes cover only these named APIs: `sys.executable`; `os.getenv`; `collections.defaultdict`;
`itertools.count` and `itertools.islice`; `heapq.heapify` and `heapq.heappop`; `bisect.bisect_left`;
`math.sqrt` and `math.ceil`; `string.digits`; `json.dumps(sort_keys=...)`; `re.sub`;
`functools.reduce`; `dataclasses.dataclass`; `typing.List[...]`; `enum.Enum`;
`argparse.ArgumentParser.prog`; `subprocess`'s capability-free entrypoint; and
`pytest`/`unittest.TestCase.__name__` imports. This is a tested API slice, not complete support for
any module. The VFS-only pytest and unittest runners currently support explicit test-file paths,
definition-order collection, plain zero-argument pytest functions, direct `unittest.TestCase`
classes, the tested assertions/skip/raises controls, and bounded generated wrappers. Fixtures,
decorated tests, plugin loading, third-party plugins, directory/package discovery, rich
parametrization, async fixtures, and unlisted runner options are explicit unsupported boundaries.

The 100-row TaskTrove mini corpus has per-row semantic provenance and compares unchanged fixtures
with checked outputs plus CPython 3.14: 99 rows are supported and one async row remains the named
frontier. The complete Python body from the public `build-system-task-ordering` reference solution
is also accepted against CPython after appending a deterministic probe. Validation includes focused
lexer/parser/compiler/protocol tests, end-to-end command/VFS/shebang/import tests, stdlib and
runner differentials, corpus differentials, full-task acceptance, and resource-stop/security tests.
The repository suite, `cargo clippy --all-targets -- -D warnings`, and `git diff --check` are required
gates; no stale aggregate test count is part of the contract.

### Known limitations of the current slice

These are deliberate boundaries, not compatibility claims hidden behind a fallback:

- integers are signed `i64`, not arbitrary precision;
- generator expressions are eagerly materialized, although ordinary generators suspend and resume;
- the invocation heap is append-only and has no garbage collector;
- descriptors, properties, and inheritance are incomplete outside the narrowly tested class,
  dataclass, enum, and unittest behavior;
- `async`/`await`, async generators, and async pytest fixtures are unsupported;
- the named stdlib modules expose only the measured and promoted API slices, not their full APIs;
- `yield` is rejected when it would cross a `try`/`finally` or `with` cleanup region.

## Decision

Build a source-compatible **Python 3.14 subset** in Rust with four explicit layers:

1. an indentation-aware lexer and hand-written parser;
2. a small, shellsim-owned bytecode format and compiler;
3. a metered stack VM with a compact Python object model; and
4. native, capability-limited implementations of the requested library and test-runner APIs.

This should not consume CPython bytecode, vendor CPython, invoke host Python, or claim general
Python compatibility. CPython bytecode is explicitly an unstable implementation detail, and
copying it would add complexity without improving source-level compatibility. The contract is the
accepted source subset plus the 23-task "pure packages" corpus. Anything outside that contract must
fail loudly and add an `unsupported` entry; it must never silently approximate an answer.

This is feasible only with that bounded contract. "All of pytest", "all of unittest", or every API
in the named standard-library modules would be a second full Python implementation. Here, "support"
means the corpus-derived API slice documented and tested function by function. The test and package
sources remain pure Python, but `pytest`, `unittest`, and the named standard-library surfaces are
small compatibility implementations, not vendored upstream packages.

## What the repository tells us

The current tree is a good host for this design:

- `src/python/mod.rs` is now the command-facing adapter for the lexer/parser/compiler/VM pipeline.
  It also owns the deliberately small VFS-only pytest and unittest runner entrypoints; unsupported
  syntax, APIs, and runner options are recorded and returned with non-zero status.
- `Environment` owns the only VFS, virtual clock/network, process state, and resource meter.
  Python should use these objects directly rather than snapshotting or mirroring them.
- Command dispatch already centralizes stdin/stdout/stderr, abstract CPU and memory charging,
  output limits, command telemetry, and nested simulated command execution.
- The VFS already provides normalized paths, symlinks, metadata, quota-atomic writes, and stable
  directory iteration. It must remain Python's only filesystem capability.
- The current Linux seccomp filter denies networking, native process execution, and some process
  creation. It is useful defense in depth, but is default-allow and is not the primary Python
  isolation boundary.
- The current baseline includes the existing shell, VFS, resource, Python language, library,
  differential, runner, and full-task acceptance suites. Validation reports are intentionally
  expressed as named gates and corpus counts rather than a brittle aggregate test count.

Repository history also contains a useful prototype at commit `1e06916`. It embedded RustPython,
then patched `open`, low-level file descriptors, `io.FileIO`, imports, `os`, `pathlib`, sockets, and
`subprocess` to keep execution in the simulated environment. It also had a small pytest driver.
That work established several requirements worth retaining: VFS package imports, isolated module
state between ordinary invocations, definition-order test collection, fixtures, and nested
`python -c` wrappers. It also demonstrates the architectural cost of starting with an interpreter
that has ambient host-facing standard-library capabilities. The proposed VM has no such capability
to patch out.

The full 23-task verifier corpus is still an external input. This worktree checks in 100 small,
per-row-provenance TaskTrove fixtures and one complete `build-system-task-ordering` solution
acceptance fixture; those are evidence for the current slice, not a claim of 23/23 task coverage.
Phase 0 below keeps the larger syntax/API inventory reproducible.

### Phase 0 evidence update

The follow-up TaskTrove/TBLite inventory in `TASKTROVE_PYTHON_INVENTORY.md` inspected all 100 public
TBLite environments and verifiers plus reference-solution Python at pinned revisions. It parsed 268
Python sources with CPython 3.14. The results reinforce the design: 61 tasks use classes, 86 use
`with`, 76 use `try`, roughly half use comprehensions or generator expressions, and 7 use `yield`.
No task uses `match`; async/await is concentrated in service-oriented tasks.

The inventory also expands the practical library floor. In addition to the reviewer-named modules,
`pathlib` appears in 61 tasks, `datetime` and `time` in 17 each, `hashlib` in 12, `random` in 10,
`csv` in 9, and `glob` in 7. These common deterministic modules should be first-tier work rather
than accidental dependencies. The source data has no `pure packages` label, so the reviewer-authored
23 task IDs are still required before creating the acceptance manifest.

## Compatibility contract

### Entrypoints

The first release should support:

- `python`, `python3`, and a new `python3.14` alias;
- `python[3.14] -c CODE [ARG ...]`, with `sys.argv[0] == "-c"`;
- `python[3.14] -` and source from stdin;
- a script stored in the shellsim VFS, including Python shebang execution;
- `python -m pytest ...` and `pytest ...` through one shared test runner;
- `python -m unittest ...` through the same collection/execution primitives; and
- the persistent shell-owned REPL, initially retaining the current one-line interaction model.

Every ordinary Python invocation gets fresh globals, modules, heap, and stdio. Only intended
machine effects (VFS changes, virtual cwd/environment changes where applicable, resource use, and
telemetry) persist. The REPL is the explicit exception and owns persistent VM state.

### Current accepted language subset

The target is ordinary task and verifier code, not every Python 3.14 novelty.

The lexer handles UTF-8 source, indentation/dedentation, comments, explicit and implicit line
joining, identifiers, bounded signed 64-bit integer/IEEE-754 float literals, and ordinary/raw/
f-string literals. A coding cookie naming anything other than UTF-8 is rejected.

The parser/compiler supports:

- literals; names; list/tuple/dict/set displays; attributes; subscripts and slices;
- unary, arithmetic, bitwise, boolean, comparison, membership, identity, conditional, and walrus
  expressions with Python precedence and chained-comparison behavior;
- calls with positional/keyword/`*` arguments and lambda expressions;
- list/set/dict comprehensions and generator expressions;
- expression, assignment, annotated assignment, augmented assignment, `del`, `pass`, `assert`,
  `return`, `raise`, `break`, and `continue` statements;
- `if`, `while`, `for`, `with`, and `try`/`except`/`else`/`finally`;
- functions, definition-time defaults, `*args`, decorators, closures, and `nonlocal`;
- basic classes, methods, independent instances, attributes, `__init__`, and the class behavior
  required by the tested dataclass/enum and unittest slices;
- imports from built-in modules and `.py` files/packages in the VFS; and
- suspended generators, subject to the cleanup-region boundary documented below.

The target remains broader than this current slice. Future extensions include keyword-only and
`**` arguments, richer descriptors/inheritance, complete generator-expression laziness, and
additional module and runner APIs. `async`/`await`, async generators, `match`, structural pattern
matching, 3.14 template strings, custom metaclasses, dynamic code-object construction,
pickle/marshal/`.pyc`, tracing/profiling hooks, native extensions, and arbitrary import hooks are
currently rejected with precise diagnostics and telemetry. Annotations are parsed/erased for the
tested cases; full Python 3.14 deferred-annotation (`annotationlib`) semantics are not implemented.

This list is a starting hypothesis. Phase 0 may move individual features in or out, but every move
must be backed by a corpus fixture and CPython differential test.

### Core data model

The current runtime types are:

`None`, booleans, bounded signed 64-bit integers, IEEE-754 doubles, strings, bytes/bytearray,
list, tuple, dict with insertion order, set/frozenset, range, slice, iterators, functions, bound
methods, generators, modules, classes, instances, exceptions, and file-like stdio/VFS objects.

All behavior routes through one protocol layer: truth testing, representation, hashing, equality
and ordering, numeric dispatch, iteration, subscription, attribute lookup, descriptors, calling,
and exception matching. Native library functions call those protocols instead of duplicating
"Python-like" behavior. Special methods are looked up on the type, not opportunistically on an
instance, matching Python's data model.

The object heap uses opaque arena IDs rather than `Rc<RefCell<_>>`. The current arena is
append-only for an invocation (there is no mark/sweep collector yet), so modeled memory limits are
the safety boundary and cyclic graphs are intentionally bounded rather than reclaimed. Immediate
`None`, bool, i64, and float values live directly in `Value`; other objects live in the heap and
carry modeled sizes. Allocation charges shellsim memory before mutation, and a failed charge raises
the VM's resource-stop signal without leaving a half-mutated object. A future collector may reuse
vacant arena slots only after its root and cycle semantics are specified and tested.

## Proposed module layout

```text
src/python/
  mod.rs                 Public shellsim adapter; replaces the existing shim incrementally
  cli.rs                 Parse Python CLI modes and construct RunRequest/sys.argv
  source.rs              SourceId, Span, line map, diagnostics, traceback locations
  token.rs               Token and keyword definitions
  lexer.rs               UTF-8, indentation, literals, f-string tokenization
  ast.rs                 Deliberately small owned AST
  parser.rs              Recursive-descent statements + Pratt expression parser
  scope.rs               Local/global/nonlocal/cell/free-name analysis
  bytecode.rs            CodeObject, constant pool, typed Op enum, verifier/disassembler
  compiler.rs            AST -> custom bytecode, jump patching, unwind metadata
  runtime/
    mod.rs               Engine and per-invocation Runtime
    value.rs             Value and ObjId
    heap.rs              Arena, roots, mark/sweep, modeled object sizes
    object.rs            Object variants and type/class/instance records
    frame.rs             Operand stack, locals/cells/globals, instruction pointer
    protocol.rs          call/attr/item/iter/hash/repr/numeric operations
    exception.rs         Exception hierarchy, traceback, unwind reasons
    vm.rs                Metered eval loop; no filesystem/process APIs
    builtins.rs          Core types, exceptions, and built-in functions
  platform.rs            The sole capability bridge to Environment, VFS, command dispatch, I/O
  imports.rs             Native registry + VFS source-package loader and sys.modules cache
  stdlib/
    mod.rs               Module registry and unsupported-API reporting
    sys.rs               argv/version/path/modules/stdin/stdout/stderr/exit
    os.rs                VFS paths, cwd, environ, stat/list/walk, safe fd facade
    collections.rs       deque/defaultdict/Counter/namedtuple/OrderedDict slice
    itertools.rs         lazy iterator constructors used by the corpus
    heapq.rs             heap operations through Python comparison protocols
    bisect.rs            bisect/insort variants including key=
    math.rs              scalar math/constants with Python domain/overflow errors
    string.rs            constants, Template, and basic Formatter behavior
    json.rs              loads/dumps/load/dump and encoder/decoder options in the corpus
    regex.rs             explicitly bounded safe regex dialect behind the `re` module
    functools.rs         wraps, partial, reduce, cache/lru_cache, cmp_to_key, singledispatch slice
    dataclasses.rs       dataclass/field/fields/asdict/astuple/replace/is_dataclass slice
    typing.rs            runtime-erased aliases plus APIs actually inspected by tests
    enum.rs              Enum/IntEnum/Flag basics integrated with class creation
    argparse.rs          common ArgumentParser/add_argument/parse_args/help/error behavior
    subprocess.rs        synchronous simulated commands; never std::process
    pathlib.rs           VFS-native Path/PurePath operations
    datetime.rs          date/time/datetime/timedelta/timezone slice
    time.rs              virtual-clock time plus deterministic formatting/parsing
    hashlib.rs           existing shellsim hash implementations exposed as Python objects
    random.rs            deterministic CPython-compatible Random slice
    csv.rs               reader/writer and DictReader/DictWriter
    glob.rs              VFS globbing
    tempfile.rs          deterministic VFS temporary paths/files
    importlib.rs         VFS module loading and spec-from-file-location slice
    io.rs                in-memory text/byte streams and Python stdio contracts
  testing/
    mod.rs               Shared discovery, invocation, result, and reporting model
    pytest.rs            pytest API/marks/fixtures and `-m pytest` adapter
    unittest.rs          TestCase assertions, discovery, suites, runner, `-m unittest`
```

`mod.rs` remains the only interface used by command dispatch. Parser, compiler, and VM depend only
inward. Only `platform.rs` may touch `Environment`; a repository lint/test should reject
`std::fs`, `std::process`, host environment, host time, and networking references elsewhere under
`src/python`.

The core interfaces should stay small:

```rust
pub fn run_python(env: &mut Environment, request: RunRequest, io: PythonIo<'_>) -> i32;

pub trait Platform {
    fn charge_cpu(&mut self, units: u64) -> Result<(), StopReason>;
    fn reserve_memory(&mut self, bytes: u64) -> Result<(), StopReason>;
    fn read_file(&mut self, path: &str) -> Result<Vec<u8>, PlatformError>;
    fn write_file(&mut self, path: &str, bytes: &[u8]) -> Result<(), PlatformError>;
    fn run_command(&mut self, argv: &[String], input: Vec<u8>) -> CommandResult;
    // cwd, directory, metadata, environment, and stdio methods omitted here
}

pub type NativeFn = fn(&mut Vm<'_>, CallArgs) -> PyResult<Value>;
```

`Platform` is not a generic host abstraction: its production implementation exposes only modeled
shellsim capabilities. The trait exists to make the security boundary reviewable and unit-testable.

## Bytecode and execution model

Use a typed Rust enum, not byte values and not CPython opcodes. About 45–60 simple operations should
cover the target: constants and stack operations; fast/cell/global/name loads and stores; attribute
and item operations; collection builders; unary/binary/compare operations; calls; function/class
creation; iteration and jumps; imports; yield/return; and raise/unwind operations.

A `CodeObject` owns operations, constants, interned names, local/cell/free-variable layouts,
argument metadata, source spans, and exception-handler ranges. The compiler first runs scope
analysis, then emits bytecode and verifies it. The verifier rejects invalid jump targets, stack
underflow, inconsistent merge heights, invalid constant/name indexes, and malformed handler ranges.
Only verified code reaches the VM.

The VM is a conventional operand-stack loop. Every instruction charges deterministic CPU fuel;
protocol loops and native library algorithms charge additional units proportional to inspected
elements. Calls check recursion depth. Container growth, source/AST/code objects, frames, modules,
and captured output reserve modeled memory. VFS mutation continues to rely on the VFS's atomic disk
quota. Resource exhaustion is a shellsim stop, not a catchable Python exception.

Control-flow unwinding uses an explicit `UnwindReason` (`Return`, `Break`, `Continue`, `Exception`,
or resource stop) plus handler metadata. This keeps `finally` and context-manager cleanup correct
without copying CPython's changing exception-stack bytecode.

## Standard-library slice

The implementation policy is "small native core, exact tested surface, loud boundary". The table
below remains the reviewed target surface and staged design; it is not a claim that every listed
API is implemented today. The measured current contract is the 18-probe gate above, with each
module intentionally partial until additional differential probes promote an API.

| Module | First supported surface | Boundary notes |
|---|---|---|
| `sys` | `argv`, `version_info`, `version`, `executable`, `path`, `modules`, stdio, `exit`, recursion limit | No tracing, audit hooks, frame access, or implementation internals |
| `os` | `environ`, `getenv`, cwd/chdir, list/stat/walk, mkdir/remove/rename/replace, path operations, simple fd I/O | VFS only; deterministic uid/pid/time values; no raw host fd or spawn APIs |
| `collections` | `deque`, `defaultdict`, `Counter`, `namedtuple`, `OrderedDict` | Add APIs only from inventory |
| `itertools` | common finite/lazy combinators (`chain`, `count`, `cycle`, `repeat`, `islice`, `tee`, zip/filter/map families, combinatorics) | Every pull is metered; unbounded iterators are safe under fuel |
| `heapq`, `bisect` | public heap/bisect/insort functions | Use VM comparisons and sequence protocols |
| `math` | constants and common scalar arithmetic/transcendentals | Match CPython exceptions and special values; differential tolerance where libm differs |
| `string` | constants, `capwords`, `Template`, basic `Formatter` | Reject unimplemented formatter grammar explicitly |
| `json` | load(s)/dump(s), hooks and common formatting options | Preserve dict order and CPython error classes/locations for the accepted input slice |
| `re` | compile/search/match/fullmatch/findall/finditer/split/sub/escape plus flags/groups needed by corpus | Backtracking-only constructs are rejected, not reinterpreted; Rust regex keeps runtime bounded |
| `functools` | decorators/helpers listed in the module tree above | Cache sizes count toward memory |
| `dataclasses` | common decorator/field/introspection/copy behavior | Frozen/order/slots/kw-only only when covered by explicit tests |
| `typing` | annotation-friendly aliases, unions, generics/protocol helpers used at runtime | Static typing has no VM effect; reflective behavior is implemented only when tested |
| `enum` | `Enum`, `IntEnum`, `Flag`, `IntFlag`, `auto`, member lookup/iteration | No exotic metaclass customization initially |
| `argparse` | common options, positional args, actions, types, choices, defaults, subparsers, usage/errors | Pin 3.14 defaults such as color/suggestion behavior in tests; default to deterministic no-color output |
| `subprocess` | `run`, `check_call`, `check_output`, `CompletedProcess`, errors, and synchronous `Popen.communicate` subset | Routes to shellsim dispatch; `python -c` nests in-process; no concurrency or host process |

The import registry also needs tiny dependency modules such as `builtins`, `types`, `abc`, `copy`,
`operator`, `keyword`, `traceback`, and `contextlib` when the corpus uses them. These must appear in
the inventory and compatibility matrix rather than arriving as invisible scope creep. SQLite,
threading, and asyncio remain separate capability decisions: the sample uses them, but they carry
substantially more semantics than deterministic helper modules.

For `subprocess`, supported arguments are lists/tuples by default; `shell=True` explicitly routes a
string through shellsim's shell parser. `cwd`, `env`, `input`, `stdin`, `stdout`, `stderr`,
`capture_output`, `text`/`encoding`, and `check` are modeled. Unsupported process-lifetime features
(`poll` races, signals, pass-through fds, process groups, real timeouts) fail loudly. A nested Python
invocation uses fresh Python globals/modules while sharing the same `Platform`, VFS, remaining fuel,
and output budget.

## Pytest and unittest contract

Do not attempt to import and run upstream pytest. Its collection, assertion rewriting, plugin
system, and transitive dependencies would dominate the interpreter project. Implement the verifier
surface directly:

- collect `test_*.py` and `*_test.py` in stable path order;
- execute module top level once and tests in source-definition order;
- collect `test_*` functions and `Test*` class methods;
- implement plain `assert` in the compiler, with useful source locations;
- support `pytest.raises`, `approx`, `fail`, `skip`, `xfail`, `param`, `fixture`, and the inventory's
  `mark.parametrize`/other marks;
- resolve fixture dependencies, scopes, generator teardown, parametrization, and `conftest.py` at
  the level exercised by the 23 tasks;
- provide faithful `tmp_path`, `tmp_path_factory`, `monkeypatch`, and output-capture fixtures where
  used, including teardown rather than no-op mutation;
- provide common `unittest.TestCase` assertions, `setUp`/`tearDown`, class hooks, discovery, suites,
  and exit status; and
- produce stable concise output while treating exit status and VFS results as the primary contract.

The TBLite sample shows that the CLI must at least accept `-rA`, `-q`, `-v`, and `--tb=short`.
Twenty-nine task runners request the third-party `--ctrf` option; the adapter must either generate
the requested report or reject the option explicitly rather than silently ignoring it.

Every skipped/xfail/unsupported marker and fixture is reported. An unknown fixture is an error, not
`None`. Plugin loading and third-party pytest plugins are out of scope unless a particular corpus
fixture is promoted into the contract.

## Safety model

The strongest property of this design is absence of ambient capability:

- User code is data consumed by a Rust parser and VM; it is never native code.
- Imports resolve only through the native registry or source files in the VFS.
- File APIs reach only `Platform` and therefore only the VFS.
- Process APIs re-enter shellsim command dispatch synchronously.
- Time and randomness are deterministic and based on modeled state.
- Network, FFI, dynamic libraries, `ctypes`, native extensions, and host process APIs do not exist.
- CPython is a development oracle only and is never linked or launched by the product binary.
- Every bytecode instruction and iterator pull consumes CPU fuel; regex work, output, source,
  generated runner wrappers, containers, and other allocations are bounded by modeled memory or
  explicit size limits. Call/expression nesting and parser/compiler nesting have explicit caps.
  Output is charged once and stops at the shellsim output budget rather than growing an unbounded
  host buffer. The current arena is append-only and has no garbage collector, so memory accounting
  is conservative by design.
- Rust panics are bugs: malformed source, malformed bytecode, user exceptions, quota errors, and
  unsupported behavior must all return typed errors.

The existing seccomp layer remains defense in depth. The Python security tests must pass with
seccomp disabled as well, proving that isolation does not depend on Linux syscall filtering.

## Validation plan

### Measured validation completed in this worktree

- `tests/fixtures/python/tasktrove_100/manifest.tsv` and `provenance.tsv` contain 100 auditable
  mini-scripts selected from the pinned TaskTrove/TBLite extraction. The differential harness runs
  every row against shellsim and CPython 3.14 when present; it currently reports 99 supported rows
  and one explicitly rejected async frontier row.
- `tests/fixtures/python/stdlib_slice/manifest.tsv` contains one deterministic probe per requested
  module/runner entrypoint. The harness currently reports 18/18 supported probes and compares exact
  stdout, stderr, and status with CPython 3.14 when available.
- `tests/tasktrove_full_acceptance.rs` extracts and runs the unmodified complete
  `build-system-task-ordering` solution from the pinned local corpus. Its output, status, and
  deterministic filesystem probe match CPython 3.14.
- Focused tests cover parser/compiler contracts, protocol behavior, language features, VFS imports,
  stdlib functions, pytest/unittest runner boundaries, output/memory/CPU limits, and unwind paths.
  Clippy with warnings denied and `git diff --check` are repository gates.

### 1. Freeze the behavioral corpus

Keep `tests/fixtures/python/tasktrove_100/manifest.tsv` and `provenance.tsv` generated from the
pinned source revision, with task ID, source path, script, expected output, and explicit status for
each row. The eventual 23-task acceptance manifest should add entrypoints, expected reward,
required modules/APIs, syntax features, and any intentional normalization. Keep full external task
data outside git if licensing or size requires it; check in minimal regressions derived from each
behavior. Record the corpus revision/hash.

The checked-in inventory tool uses CPython 3.14's `ast`, import tracing, and source inspection to
report:
syntax nodes, imports, accessed module attributes, subprocess shapes, fixture/mark usage, exit code,
stdout/stderr, and the final file manifest. Dynamic tracing is advisory; static scans and successful
differential execution remain the truth.

### 2. Layer tests

- **Lexer:** table tests for indentation, continuations, Unicode, numbers, strings, and bad input;
  compare token kinds/spans with CPython's `tokenize` where meaningful.
- **Parser:** canonical AST snapshots for every accepted construct; compare success/failure and a
  normalized AST with `python3.14 ast.dump(..., include_attributes=True)`.
- **Scope/compiler:** golden symbol tables and human-readable bytecode; bytecode verifier property
  tests for stack heights, jumps, indexes, and handlers.
- **VM:** one-semantic-rule tests for evaluation order, side effects, closure capture, argument
  binding, descriptors, iteration, exceptions, `finally`, and generator teardown.
- **Libraries:** a table per public supported API containing normal, boundary, mutation, exception,
  repr, and unsupported cases.

### 3. CPython 3.14 differential harness

The current `stdlib_slice` and `tasktrove_100` fixtures plus their Rust integration harnesses are
the first differential suite. Extend those fixture directories (or add
`tests/python/differential/cases/*.py`) in development/CI jobs
that provide `python3.14`, each case runs once in CPython 3.14 and once in shellsim. Compare:

- exit status;
- stdout and stderr bytes (with narrowly documented traceback/path normalization);
- exception type and message for API contract tests;
- selected object results serialized by a tiny oracle helper; and
- a normalized VFS manifest: path, kind, mode, contents, and relevant metadata.

Set `PYTHONHASHSEED=0`, locale, timezone, terminal color variables, and argv identically for the
oracle. Never broadly normalize ordering, float text, or error messages merely to make tests pass.
For transcendental math only, use explicit per-case tolerances. CPython-dependent tests should skip
cleanly on contributor machines without 3.14, while a required CI lane runs them.

### 4. Property, fuzz, and adversarial tests

- Differential property tests for integer arithmetic, slices, sequence operations, dict behavior,
  heap/bisect invariants, JSON round trips, and the accepted regex dialect.
- Parser and bytecode-verifier fuzz targets: arbitrary bytes must never panic, hang, or reach the
  VM as unverified code.
- Resource attacks: infinite loops/iterators, recursion, enormous integers/containers, cyclic
  graphs, output floods, pathological accepted regexes, and repeated imports.
- Capability attacks with seccomp disabled: `/etc/passwd`, traversal/symlinks, host environment,
  sockets, `ctypes`, native imports, `os.system`, `subprocess`, import-hook mutation, and attempts to
  fabricate code objects. Assert no host effects and precise unsupported telemetry.
- Determinism: repeat each accepted program and compare status, output, VFS, telemetry, and modeled
  resource usage exactly.

### 5. End-to-end acceptance

The release gate is not merely "the verifier exits zero." For every one of the 23 tasks:

1. run the same oracle/solution and verifier under CPython 3.14 and shellsim;
2. compare exact reward, test exit status, normalized output, and relevant filesystem state;
3. require no unexpected unsupported feature or partial/no-op command;
4. run twice to establish determinism; and
5. retain any discovered divergence as a minimized regression test.

Also require all existing shellsim tests, Python layer tests, differential tests, adversarial tests,
`cargo fmt --check`, clippy, and `git diff --check` to pass. Track coverage as a matrix of
task × syntax feature × module API, not as a single optimistic percentage.

## Delivery sequence

Each phase ends in a usable vertical slice and a reviewable commit.

### Phase 0 — inventory and harness

Restore a modernized version of the old differential harness, obtain/pin the 23 tasks, produce the
feature/API matrix, add `python3.14` command registration, and preserve current shim tests as
compatibility fixtures. No semantic implementation claim is made yet.

### Phase 1 — source pipeline and scalar VM

Land source/span diagnostics, lexer, expression/statement parser, AST, scope analysis, typed
bytecode, verifier, heap, frames, scalar/container values, builtins, and a metered eval loop. Replace
the existing ad-hoc expression splitter for supported scalar `-c` programs while keeping a clear
unsupported fallback during migration.

### Phase 2 — real programs

Add functions/closures, argument binding, imports from the VFS, exceptions/with/finally,
comprehensions, generators, classes, descriptors, script/stdin execution, tracebacks, and isolated
per-invocation module state. At this point small multi-file Python packages should run.

### Phase 3 — platform and named libraries

Land `sys`/stdio, `os`, VFS files, and `subprocess` first; then pure computation modules in small
groups with differential tables. Add tiny dependency modules only through the reviewed inventory.

### Phase 4 — verifier runners

Implement shared test discovery, pytest compatibility, fixtures/teardown/parametrization, and
unittest compatibility. Drive this phase by minimized examples from the 23 verifiers.

### Phase 5 — corpus closure and hardening

Run the full differential suite, close gaps one minimized behavior at a time, fuzz, run the
seccomp-disabled security suite, document the final supported API matrix, and update README trust
language. The phase completes at 23/23 exact reward matches with no hidden fallback.

## Review checkpoints and likely risks

1. **After Phase 0:** approve the actual scope matrix. This is the main defense against accidentally
   promising all of Python/pytest.
2. **After the bytecode skeleton:** review stack/unwind design and allocation accounting before
   library breadth makes changes expensive.
3. **After classes/descriptors:** verify dataclass/enum/test-class requirements; these stress the
   object model most.
4. **After `os`/`subprocess`:** perform a focused capability audit before adding more APIs.

The largest correctness risks are descriptors and multiple inheritance, `finally` during nonlocal
control flow, generators/fixture teardown, Python-compatible repr/error details, regex dialect
differences, and reflective behavior in `typing`/dataclasses/enums. The largest schedule risk is an
unstated dependency in the 23 verifier files. The matrix and loud unsupported contract make those
risks visible early.

## Sources

- [Python 3.14 language reference](https://docs.python.org/3.14/reference/index.html) — source-level
  syntax and core semantics.
- [Python 3.14 full grammar](https://docs.python.org/3.14/reference/grammar.html) and
  [CPython parser internals](https://github.com/python/cpython/blob/3.14/InternalDocs/parser.md) —
  oracle grammar and PEG/error-rule context.
- [CPython compiler internals](https://github.com/python/cpython/blob/3.14/InternalDocs/compiler.md)
  — parser, AST, symbol table, compiler, code object, and eval-loop stages.
- [Python `dis` documentation](https://docs.python.org/3.14/library/dis.html) — explicit warning that
  CPython bytecode is an unstable implementation detail.
- [Python 3.14 data model](https://docs.python.org/3.14/reference/datamodel.html) — attributes,
  descriptors, class creation, special methods, and object protocols.
- [Python 3.14 standard library](https://docs.python.org/3.14/library/index.html) — behavioral oracle
  for the requested module slices.
- [Python 3.14 argparse documentation](https://docs.python.org/3.14/library/argparse.html) — notable
  3.14 additions and CLI behavior.
- [pytest plugin architecture](https://docs.pytest.org/en/stable/how-to/writing_plugins.html) — why
  upstream pytest is materially broader than the proposed verifier compatibility runner.

## Recommendation

Proceed, beginning with Phase 0 and a narrow Phase 1 vertical slice. Do not revive RustPython and do
not start by implementing modules. The source pipeline, object protocols, resource accounting, and
corpus differential harness are the foundation; library functions built before those contracts are
stable would be fast to write and expensive to trust.
