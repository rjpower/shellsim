# Wasmtime fuel-resume overlay

The source overlay adds `CallHook::FuelResume` after a fuel suspension and before
the guest continues. Shellsim uses the exclusive Store context at this boundary
to replay published threaded modules and callback slots. Host-return replay is
handled separately. The pinned crate archive, patch inputs and patch digest are
recorded in `recipe.json`; `build.py` reproduces the vendored source.

Python source distributions include `vendor/wasmtime` and compile that path
dependency. The extracted source distribution must build with Cargo's locked
resolution before publication. The upstream Wasmtime license is retained.

Cargo registry publication is disabled for Shellsim while this overlay remains
a path dependency. Cargo normalizes a published path dependency to its registry
version, which would omit the required resume hook. Publishing a separately
versioned patched runtime and updating the dependency is a distinct release
step; no such release is part of this port.
