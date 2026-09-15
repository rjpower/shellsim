# PyPI packaging design

## Decision

Publish one `shellsim` distribution containing an ABI-stable PyO3 extension and a small typed
Python facade. The facade exposes fresh and persistent simulated environments without starting a
host subprocess. Keep PyO3 in a dedicated adapter crate so the existing Rust library, binaries,
features, and security boundary do not depend on Python.

The extension must never call `sandbox::apply()`. That function installs a process-wide,
irreversible Linux seccomp filter and is appropriate only for shellsim's standalone binaries.
Embedding must not change the host Python process's ability to create sockets or subprocesses.
The extension relies on the Rust library's capability-free execution boundary; the binary-only
kernel backstop is not part of the wheel's contract.

The first public API should support this shape:

```python
import shellsim

result = shellsim.run("printf 'hello\\n'")
assert result.returncode == 0
assert result.stdout == b"hello\n"

environment = shellsim.Environment(cpu=100_000, output=16_384)
environment.write_file("/work/main.py", b"print(6 * 7)\n")
result = environment.run("python3.14 /work/main.py")
assert result.stdout == b"42\n"
```

`Environment.mount(path, destination="/work")` may import an explicitly selected trusted host
tree through the existing bounded, symlink-rejecting ingestion path. It returns a `MountResult`
that reports copied files and skipped directory names. Simulated commands must still receive no
ambient host filesystem, process, network, environment, or clock access.

The distribution also exposes a `shellsim` console entry point. It supports one-shot `-c` actions,
piped shell source, and a persistent terminal session. `--root DIR` copies an explicitly trusted
host tree into `/work` before execution; the copy is a disposable VFS snapshot and has no write-back
path to the host.

## Repository precedent

The current Marin-community repositories use several build backends, but the relevant boundary is
stable:

- [Marin's workspace metadata](https://github.com/marin-community/marin/blob/main/pyproject.toml)
  uses PEP 621 metadata and uv dependency groups/workspaces. It deliberately excludes native
  Maturin packages from the Python workspace because each native package owns its build backend.
- [Marin dupekit's native package](https://github.com/marin-community/marin/blob/main/lib/dupekit/rust/pyproject.toml)
  and [finelog's native package](https://github.com/marin-community/marin/blob/main/lib/finelog/rust/pyproject.toml)
  use Maturin with a thin PyO3 crate, an explicit module name, and source-aware uv cache keys.
  Dupekit uses ABI3 so one wheel covers the supported CPython versions.
- [Marin's release workflow](https://github.com/marin-community/marin/blob/main/.github/workflows/marin-release-libs-wheels.yaml)
  separates planning, platform builds, artifact validation, and publishing. Publishing uses a
  protected GitHub environment and PyPI OIDC rather than a long-lived token.
- [Kitoken's Python package](https://github.com/marin-community/kitoken/blob/main/packages/python/pyproject.toml)
  is the closest standalone Rust precedent: Maturin, PyO3, ABI3, explicit classifiers, type stubs,
  and separate wheels for Linux, macOS, and Windows.
- [Loom's native adapter](https://github.com/marin-community/loom/blob/main/crates/weaver-py/pyproject.toml)
  isolates its PyO3 `cdylib` from the main Rust workspace and uses an ABI3 floor. Its separate
  [pure-Python facade](https://github.com/marin-community/loom/blob/main/python/weaver-loom/pyproject.toml)
  is stdlib-only and independently testable.
- [Levanter](https://github.com/marin-community/levanter/blob/main/pyproject.toml),
  [Haliax](https://github.com/marin-community/haliax/blob/main/pyproject.toml), and
  [Draccus](https://github.com/marin-community/draccus/blob/main/pyproject.toml) confirm the shared
  use of PEP 621 metadata, declared Python floors, package data/type metadata, dependency groups,
  and source distributions. Draccus's release checks that a `v*` tag matches the project version
  before building and publishes with OIDC.

The `shellsim` project name returned no project from PyPI on 2026-09-15. Name ownership still has
to be established by the first trusted publication.

## Package layout

Use the repository root as the Python project and keep the native adapter below it:

```text
pyproject.toml
python/
  shellsim/
    __init__.py
    _api.py
    py.typed
  native/
    Cargo.toml
    src/lib.rs
python_tests/
  test_api.py
```

The root `pyproject.toml` uses `maturin` as its PEP 517 backend, `python` as its mixed-project
source directory, and `shellsim._native` as the extension module. `python/native/Cargo.toml` is a
standalone Cargo workspace and depends on the repository's `shellsim` crate by path. It builds a
`cdylib` with PyO3's `extension-module` and `abi3-py39` features. Python 3.9 is a reasonable floor:
it matches the broadest maintained sibling packages and does not constrain the Rust engine.

Maturin reads the distribution version dynamically from the adapter manifest. The root Rust crate
and adapter versions remain equal and the release workflow checks both against the tag. The root
crate has an explicit Cargo package include list so Maturin vendors only the Rust sources,
manifests, lockfile, README, and license needed by the adapter's path dependency. The adapter owns
one checked-in lockfile, and CI builds it with `--locked`. Release validation must unpack the sdist
outside the checkout, build a wheel from that unpacked tree, install it, and run a smoke test.

The Python facade owns the public names and converts the extension's narrow serialized result into
frozen dataclasses. This keeps PyO3 conversion code small and makes the Python contract easy to
type-check. The native object owns one mutex-protected Rust `Environment`; callers may use it from
different Python threads, while actions on the same environment remain serialized.

Each action detaches from the Python interpreter, moves execution to a scoped Rust worker with an
explicit 8 MiB stack, and rejoins before returning. This prevents the Rust parser/VM recursion
limits from depending on a host Python thread's smaller platform stack and allows unrelated Python
threads and signal delivery to proceed. The adapter catches Rust unwinds around the action and
converts them to `SimulationError`; panics must not escape as PyO3 `PanicException` values.

## Initial API contract

- `Limits` is a frozen dataclass with the Rust defaults for CPU, memory, disk, and output.
- `RunResult` contains `returncode`, byte-preserving `stdout` and `stderr`, `stop_reason`, `limits`,
  cumulative `usage`, per-command usage, cost-model version, unsupported capability records, and
  the modeled no-op/partial-command sets used to detect false-positive package tests.
- `Environment(limits=None, *, cpu=None, memory=None, disk=None, output=None)` rejects booleans,
  negative values, values above `u64`, and conflicting aggregate/per-field limit arguments.
- `Environment.run(source, stdin=b"")` preserves simulated state and cumulative fuel across calls.
- `Environment.write_file`, `read_file`, and `mkdir` operate only on the VFS. Relative paths use
  the simulated current directory.
- `Environment.mount` is an explicit harness operation using `host_ingest::mount_host_tree`; it
  rejects host symlinks, remains subject to the configured disk limit, and reports the fixed skip
  list. The input is trusted harness data: the host walk is not a defense against a concurrently
  mutating adversarial tree.
- `Environment.terminated` reports resource-terminal state. A run after CPU, memory, or output
  exhaustion returns immediately with the same terminal outcome, matching the Rust API.
- `run(...)` creates a fresh environment and executes one action.

The initial release does not claim full Bash, Unix, CPython, pytest, or package-installation
compatibility. Existing unsupported reporting remains part of every result. It also does not expose
arbitrary Rust internals or callbacks from the simulator into Python.

## Build, test, and release

Add Python contract tests for fresh execution, persistent state, byte-preserving stdio, VFS file
operations, trusted mounting and rollback, invalid limits, resource exhaustion/reuse, deep input,
and unsupported/no-op/partial reporting. A Linux non-interference test verifies that importing and
running shellsim does not prevent the host process from opening a socket or launching a subprocess.
Build a wheel locally, install it into a clean virtual environment, and run the contract suite
against the installed artifact. Extend the existing lint gate to format and lint the adapter and
syntax-check the facade while preserving the complete Rust test suite.

Add a tag-triggered `publish-python.yml` workflow with these stages:

1. Verify that `vX.Y.Z` matches the root and adapter Cargo versions and that the tagged commit is
   reachable from `main`.
2. Build ABI3 wheels for Linux x86-64 and aarch64, macOS x86-64 and arm64, and Windows x86-64.
3. Build one source distribution and run artifact checks plus an installed-wheel smoke test.
4. Exercise the same artifacts against TestPyPI, then publish only after every artifact succeeds,
   through a protected `pypi` environment with `id-token: write` and
   `pypa/gh-action-pypi-publish`. Pin third-party actions to commit SHAs and do not cancel an
   in-progress release.

The first change can add the build and release contract without publishing anything. Registering a
pending trusted publisher on PyPI and TestPyPI, approving the protected environments, and pushing a
release tag remain explicit owner operations. They must happen before the first tag-driven build so
the currently unclaimed name is not exposed to a preventable publication race.

## Implementation status

The initial package, native adapter, facade, contract tests, CI job, and tag-driven release workflow
are implemented in this change. Fable's review of revision 1 identified the embedding, sdist,
telemetry, mount-reporting, terminal-state, and release issues reflected above; the implementation
includes those changes.

Local validation on Linux x86-64 built `shellsim-0.1.0.tar.gz`, rebuilt an ABI3 wheel from that
sdist, installed the wheel into a clean Python 3.13 environment, passed all 13 Python contract tests,
and passed strict Twine checks for both artifacts. The repository lint gate and complete Rust test
suite also pass. Cross-platform wheel execution and both registry stages remain workflow checks;
no package or external release state was created locally.
