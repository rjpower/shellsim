# HTTPX

Stages the official wheel and its existing dependency recipes. The declared probe
uses upstream `MockTransport` for redirects, cookies, JSON bodies and HTTP errors.
`tests/probes/asgi_fastapi.py` exercises upstream `ASGITransport` with the existing
FastAPI and sniffio ports in the integrated guest. Immediate coroutines avoid a
socket-backed event loop; serving and suspended operations are outside this probe.

HTTPX and HTTPcore import `ssl` unconditionally. Their guest probes require a
CPython SSL provider, which this checkout does not supply. The wheels are unchanged.
