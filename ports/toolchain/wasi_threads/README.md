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
the static threaded ABI's supported behavior. Threaded dynamic v3 remains
separate unfinished work and is not enabled by this recipe.
