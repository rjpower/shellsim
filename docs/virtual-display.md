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

## External Doom end-to-end probe

The ignored `tests/doom_probe.rs` test builds Doomgeneric from source with the virtual TinyCC
toolchain, links a small shellsim platform adapter, loads a separately supplied Freedoom WAD,
yields 640×400 frames, and checks that an Escape key injected between frames changes the image. Neither the
GPL engine nor game data is included in the repository. The test harness alone imports the
trusted host inputs into the VFS before simulated execution; the guest cannot reach those host
paths.

The proven inputs were:

- [Doomgeneric](https://github.com/ozkl/doomgeneric) commit
  `dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284`, with
  `SHELLSIM_DOOM_SOURCE` pointing to its `doomgeneric/` source directory.
- [Freedoom 0.13.0](https://github.com/freedoom/freedoom/releases/tag/v0.13.0)
  `freedoom1.wad`, SHA-256
  `7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d`, with
  `SHELLSIM_DOOM_WAD` pointing to that file.

Run the opt-in probe with both variables set:

```sh
SHELLSIM_DOOM_SOURCE=/path/to/doomgeneric/doomgeneric \
SHELLSIM_DOOM_WAD=/path/to/freedoom1.wad \
cargo test --test doom_probe -- --ignored
```

To play with a separately obtained WAD, run the local browser demo from this PR:

```sh
cargo run --release --example doom_player -- /path/to/doomgeneric/doomgeneric /path/to/freedoom1.wad
```

The first launch compiles Doomgeneric inside shellsim's VFS and may take about a minute. Open the
loopback URL printed by the example. Arrow keys move, Ctrl fires, Space uses, Shift runs, and Esc
opens the menu. The browser only sees copied RGBA frames and sends bounded key events to a host
demo process; the guest has no host network access. The demo binds an ephemeral `127.0.0.1` port
and exits when its Stop button is pressed. The external [Freedoom release](https://github.com/freedoom/freedoom/releases/tag/v0.13.0)
provides a playable WAD; shellsim does not download or bundle it.

The probe and demo select Doomgeneric's console-error path instead of its optional Zenity
`system()` call. The platform adapter still uses a deterministic, adapter-local tick
approximation. Audio and networking are outside this single-player proof. The original Doom/Quake
engines and assets are not bundled.
