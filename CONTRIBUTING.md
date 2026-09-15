# Contributing to shellsim

Shellsim accepts focused changes that improve a concrete simulated behavior, close a known
compatibility gap, or strengthen a safety boundary. Please open an issue before a broad redesign
or a speculative refactor.

Read [docs/implementation.md](docs/implementation.md) before adding a command or machine capability,
and [docs/python.md](docs/python.md) before changing the Python runtime or native modules.

This workflow adapts the contribution discipline used by
[Marin](https://github.com/marin-community/marin/blob/main/docs/dev-guide/contributing.md) and its
[engineering guidelines](https://github.com/marin-community/marin/blob/main/docs/explanations/guidelines.md)
to this repository's Rust runtime and deterministic-simulation constraints.

## Setup

Install rustup and Python 3. The checked-in `rust-toolchain.toml` pins Rust and installs `rustfmt`
and `clippy`; the repository wrappers have no third-party Python dependencies:

```sh
make check
```

To opt into the repository's pre-commit lint hook:

```sh
make setup_pre_commit
```

The hook only installs local Git configuration. It does not modify global hooks.

## Linting

`infra/pre-commit.py` is the canonical lint entrypoint used both locally and in CI:

```sh
make lint
```

It checks formatting, runs clippy over every target and feature with warnings denied, builds
rustdoc with warnings denied, and rejects staged or unstaged whitespace errors. Apply safe
formatting fixes with:

```sh
make format
```

The `--changed-files` and `--all-files` spellings are both accepted to match Marin's workflow.
Rust module-wide formatting and semantic checks intentionally inspect the whole small crate.

## Testing

Run the narrowest useful test while editing:

```sh
cargo test --test python_stdlib_differential
```

Before requesting review, run all safe tests through the same entrypoint as CI:

```sh
make test
```

Changes to the Python package should also build and test the installed artifact rather than import
from the source tree:

```sh
uv build
uv venv /tmp/shellsim-wheel-test
uv pip install --python /tmp/shellsim-wheel-test/bin/python dist/*.whl pytest
/tmp/shellsim-wheel-test/bin/python -m pytest python_tests
```

The safe suite is the full shellsim suite because it is local, deterministic, and does not require
Docker, a cluster, or network access. Reference-implementation differentials skip their live
comparison when the named reference binary is unavailable, while checked fixtures still run.

Use the following testing layers for new behavior:

1. Put small algorithm and invariant tests beside the implementation.
2. Put cross-module and command behavior in `tests/`.
3. Compare compatibility surfaces with CPython or coreutils where practical.
4. Add a reduced real-task fixture when a TaskTrove case exposes the gap.
5. Test malformed input, limits, and unsupported behavior as well as the success path.

Never make a test depend on host wall time, locale, network state, filesystem contents, or hash-map
iteration order.

## Code and documentation

- Start each production module with a high-level `//!` description of its role and constraints.
- Document non-obvious public APIs and functions. Explain invariants and tradeoffs rather than
  translating the body into prose.
- Prefer explicit typed state and small interfaces. Extract shared machinery when it clarifies a
  boundary or after the pattern repeats.
- Keep simulated input capability-free and resource-metered. Unsupported behavior must fail
  loudly instead of reaching the host.
- Add direct tests for tricky logic.
- Keep documentation factual and concise. Record limitations as current behavior, not promises.

## Issues and pull requests

An issue should state the problem, relevant evidence, and a concrete definition of done. Include
commands, stack traces, task IDs, or compatibility examples where they help another contributor
reproduce the problem.

A pull request should lead with what changed and why. Before opening it:

1. Run `make format`.
2. Run `make lint`.
3. Run `make test`.
4. Review the diff for unrelated edits and accidental capability expansion.
5. Report the exact validation results and remaining caveats.
6. Link the issue with `Fixes #NNN` or `Part of #NNN` when applicable.

Use an imperative title of at most 72 characters, optionally beginning with a `[scope]` prefix.
Do not use Conventional Commit prefixes such as `feat:` or `fix:`. Write the body so it can stand
alone as the squash-commit message: changed behavior first, then motivation, evidence, and material
caveats. Avoid file-by-file narration and boilerplate test headings.

Project prose should be plain, factual, and direct. Support claims with named evidence and state
limitations clearly. Avoid hype, generic significance claims, filler, decorative headings, emoji,
stock contrast constructions, synonym rotation, and em-dash asides.
