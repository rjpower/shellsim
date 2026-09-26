# Virtual display and interactive Wasm sessions

Shellsim exposes one metered RGBA8888 framebuffer and a bounded key-event queue. The virtual
display does not open a host window, read a host keyboard, or grant a guest ambient device
access. Rust programs use the typed device/syscall boundary; Wasm programs import the following
`shellsim` functions. All integers are 32-bit, and pointers refer only to guest linear memory.

| Import | Result | Behavior |
| --- | --- | --- |
| `display_open(width, height, format)` | Positive handle, or negative error number | Format `1` is tightly packed R, G, B, A bytes; dimensions must be 1–2048. Only one process owns the display at a time. |
| `display_present(handle, pointer, length, stride)` | `0` or error number | Copies one complete frame. `length` must equal `width * height * 4`, and `stride` must equal `width * 4`. |
| `input_poll_key(handle, result_pointer)` | `0`, `EAGAIN`, or error number | On success writes two little-endian `u32` values: guest-defined key code and pressed flag. `EAGAIN` means the queue is empty; no event is invented. |
| `display_close(handle)` | `0` or error number | Releases ownership but retains the last frame for inspection. |

The environment injects events with `Environment::inject_key` and reads the last frame through
`Environment::display.frame()`. An event can be queued before a guest starts. The queue holds at
most 256 events. Frame bytes and queued events count against modeled memory; frame copies charge
CPU. Bad geometry, pointers, ownership, and resource limits fail explicitly. The current ABI is
an experimental shellsim extension, not a WASI standard.

## Host-driven interaction

`WasmSession` preserves guest execution between calls to `poll`. A frame presentation yields to
the host, which can inspect the copied frame, inject bounded key events, then resume. The host can
also stop the session at a frame boundary. The session owns its environment until it stops; live
Wasmtime state is never copied by `Environment::clone()`.

A session buffers the guest's standard streams instead of using process descriptors; a Wasm
executable started from the shell runs as a scheduled process instead. The interface supplies a
display and keys, not audio or networking.
