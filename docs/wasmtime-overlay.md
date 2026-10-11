# Wasmtime resume hook

Shellsim vendors the Wasmtime 49.0.0 crate to add a resume checkpoint required by
threaded CPython dynamic loading. This was introduced in
[PR #185](https://github.com/rjpower/shellsim/pull/185), before the Buildomatic
rollout. The vendored crate is a Cargo path dependency; Wasmtime's other crates,
including Cranelift, remain registry dependencies.

Each guest thread has its own Wasmtime Store. A process shares compiled modules
and allocation addresses, but Wasmtime instances, functions, tags, and globals
belong to one Store. When a thread loads a shared library or publishes a callback,
other threads must install their own bindings before using it. Shellsim performs
that replay while it has exclusive access to a suspended Store.

Upstream's host-call hooks let Shellsim synchronize on host calls and returns.
They do not provide the corresponding hook when a compute-bound thread resumes
after a fuel yield. Such a thread can otherwise continue with stale bindings.
Polling an async call from the scheduler does not supply the exclusive mutable
Store context needed to repair those bindings before generated code continues.

The sole runtime patch adds `CallHook::FuelResume` and invokes it after an actual
fuel suspension, before execution continues. It also handles the new event in
Wasmtime's profiling hook and protection-key bookkeeping. Shellsim's thread hook
uses it to replay published modules and callback slots and check pending
termination. Module initialization and allocation remain coordinated by Shellsim;
the hook does not add a separate guest loader or host capability.

The exact upstream archive, SHA-256, patch, and input-file identities are recorded
in [the overlay recipe](../ports/toolchain/wasmtime_resume/recipe.json). The
[patch](../ports/toolchain/wasmtime_resume/fuel-resume.patch) changes three runtime
files. The source-tree builder reproduces the overlay and omits the upstream
`Cargo.toml.orig` packaging file; the upstream license remains in
`vendor/wasmtime/LICENSE`. Wasmtime upgrades must reapply this patch and exercise
threaded dynamic loading, callback publication, and fuel-resume behavior.

Python source distributions include this crate and compile the path dependency.
Shellsim currently disables Cargo registry publication: Cargo would normalize
the path dependency to upstream Wasmtime and lose the required hook. A separately
published patched crate, or an upstream hook with this contract, is needed before
that restriction can be removed. See the
[overlay port documentation](../ports/toolchain/wasmtime_resume/README.md).

Wasmtime's parallel compilation and native artifact cache are separate features;
they do not explain vendoring. Their settings and measured behavior are documented
in [guest Clang compilation and loading](guest-clang-performance.md).
