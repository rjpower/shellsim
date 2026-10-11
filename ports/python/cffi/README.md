# CFFI

This port builds upstream CFFI against the scalar Wasm libffi provider. The guest
probe covers allocation, pointer access, a virtual process symbol call, and a
Python callback through a libffi table slot. Library lookup uses Shellsim's
virtual loader. Callback storage never needs executable memory mappings.

Aggregate and variadic foreign calls inherit libffi's unsupported frontier.
On-demand host compilation through `FFI.verify()` is outside this port's runtime
scope; compilation belongs in the ports build workflow.
