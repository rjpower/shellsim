# WASI pthread toolchain

`build.py` builds pinned LLVM/LLD and wasi-libc with the scheduler ABI
`shellsim_threads_v1`. The opt-in LLD serial memory initialization flag replaces
raw atomic waits in initialization. The libc patch sends pthread waits and
notifications to Shellsim's cooperative scheduler and virtual clock.
Production recipes contain source archives, patches, SDK binaries and runtime
files, build drivers and shared helpers. Every production input participates
in cache identity; moving a driver requires a new toolchain artifact manifest.

```sh
uv run --no-project -m ports.toolchain.wasi_threads.build --help
```

The builder limits compilation to eight jobs, linking to one job, each command
to one hour and each child address space to 12 GiB. Build logs, commands, host
tool versions and output hashes are recorded in `toolchain-manifest.json`.
It builds no pthread test programs or standalone runner.

Fixture programs, their builder, standalone diagnostic runner and proof recipe
live in `tests`. Their manifest records the production toolchain recipe and
fixture provenance separately. See `tests/README.md` for guest acceptance and
the static threaded ABI's supported behavior. Threaded dynamic v3 uses the
separate dynamic sysroot below and the matching CPython/runtime cohort; the
static recipe does not enable dynamic loading.

## Threaded dynamic sysroot

`dynamic.py` builds the separate `dynamic-recipe.json` product. It verifies and
extracts the pinned SDK and wasi-libc archives, applies scheduler wait/notify and
worker stack-readiness and parent TLS restoration patches, and overlays the rebuilt C runtime onto the
SDK sysroot. The SDK's C++ headers and threaded exception runtime archives stay
in that sysroot. Target compilation uses the isolated target environment.

Pass `--sdk-archive`, `--libc-archive`, `--llvm-prefix`, `--cmake`, `--ninja`, and
`--work`. The compiler prefix must be the normal `llvm/threaded.py` product;
the complete compiler manifest and exact threaded compiler recipe are checked.
Each build requires a fresh output directory. `prefix/manifest.json` records
the recipe, compiler identity, host build tools, commands, and hashes of every
sysroot and license file. A driver or input change requires a new product.

Consumers compile with `--target=wasm32-wasip1-threads -pthread` and
`--sysroot=<prefix>/sysroot`. This product supplies the
`shellsim-wasi-sdk34-cpython3137-threads-v3` cohort and scheduler namespace
`shellsim_threads_v2`; it cannot replace v1 or nonthreaded v2 artifacts.
It builds no interpreter or acceptance fixtures. Production threaded loading
and upstream interpreter acceptance are verified separately.

The parent TLS patch restores optional dynamic TLS global relocations after
`__copy_tls` initializes a child TLS block and restores the parent TLS base.
The relocation function is weak, so ordinary static links without dynamic TLS
relocations remain supported.

The dynamic sysroot also builds musl's real `pthread_atfork` registry, adapted to
its current strong-lock primitives. Registration allocates a synchronized
callback list and retains musl's `ENOMEM` behavior. The internal fork dispatcher
calls prepare handlers in reverse registration order and parent/child handlers
in registration order. CPython's verified relink refreshes the public libc
symbol inventory so newly admitted APIs are retained in its canonical runtime.

Guest process creation uses spawn/exec; it does not invoke fork callbacks.
Guest `fork` remains unsupported. The threaded CPython atfork probe registers
real callbacks, checks that subprocess creation does not invoke them, and calls
the musl dispatcher explicitly to verify each callback order.
