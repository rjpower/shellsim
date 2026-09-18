# Requested stdlib differential probes

These deterministic probes cover one small, observable API from every module in the reviewed
Python 3.14 stdlib slice, plus `pytest` and `unittest` entrypoint probes. The source is intentionally
small enough to audit and is executed unchanged by CPython 3.14 and shellsim.

`supported` is the current shellsim compatibility contract. `frontier` is a valid CPython probe
whose module or API is not yet in the emulator; shellsim must reject it loudly. The harness first
checks `sys.implementation.name` and the major/minor version, and only uses the executable as an
oracle when it is actually CPython 3.14. The pytest probe has an `any` host status because pytest
is a third-party package; its checked output is empty and deterministic either way.
