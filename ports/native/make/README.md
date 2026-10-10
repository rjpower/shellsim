# GNU Make 4.4.1 for the virtual POSIX process runtime

The recipe builds upstream GNU Make with SDK 34 against the versioned
`shellsim_posix_v1` facade. Target programs execute only in Shellsim. Host configure,
make and C generators are recorded by executable hash; source, SDK, patches,
facades and license inputs are pinned in the artifact manifest.

The source patch fixes flexible `dirent.d_name` allocation, selects the existing
SDK-compatible two-argument main, and guards unavailable signal-mask/callback
facilities. NLS and Guile are disabled. Ordinary parallel recipes use real virtual
spawn/wait and work with `-j2`. Recursive jobserver coordination and asynchronous
signal callbacks remain unsupported. This package supplies make; combine its
graph recipe with the [guest Clang port](../../toolchain/llvm/GUEST.md) for C and
C++ compilation, linking and archiving inside the guest.

Configure's synchronous-posix-spawn cache describes the measured non-null-pid
spawn path used by make. The upstream probe passes a null pid, which this bounded
facade rejects. Other cache entries describe missing SDK signal facilities.

Build with `uv run --no-project --python 3.13 python -m ports.native.make.build
SOURCE_ARCHIVE SDK WORK --jobs 4`. Artifacts export the executable and complete license
notices. Account lookup reads bounded guest `/etc/passwd`; it never imports host
identity. Temporary files use guest random bytes, atomic exclusive creation and
mode 0600 before virtual umask.

Measured guest proofs include two concurrent one-second recipes completing in one
virtual second and a remade included Makefile restarting make with
`MAKE_RESTARTS=1`. Public catalog installation supplies the same verified executable.

## Graph build

`graph-recipe.json` builds GNU make through the common configure/make adapter.
It explicitly depends on the host LLVM compiler, threaded target SDK and
`native/shellsim-posix` library. Process and descriptor code is compiled once
by that library port. This recipe publishes make as a guest build tool; its
port-local shell check runs parallel jobs and checks propagation of a failed
recipe through the public release installation API.

The graph recipe is a distinct threaded build from the earlier standalone
`recipe.json` profile. Its successful build alone does not establish guest
execution; use `--output ... --check` to build and run the declared check.

On WASI, Make retains its bundled GNU option parser and glob implementation,
including directory callbacks used by wildcard expansion. The graph patch gives
these functions and parser state private names so they coexist with the canonical
executable's exported libc implementations. Configure still rejects the SDK glob
as a GNU glob provider. The shell check exercises long options, wildcard includes,
parallel recipes and recovery after an invalid option.
