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

## External Doom package

`examples/build_doom_package.py` puts Doomgeneric source, a Freedoom IWAD, the pinned virtual
TinyCC toolchain, a WASI sysroot, and shellsim's display adapter into one `.shl` blob. The
package's sole entrypoint extracts the toolchain, compiles the engine inside shellsim, and runs
the resulting Wasm program. It requests real-time clock mode for interactive play. The host may
instead instantiate it with virtual time for deterministic probes. Neither the GPL engine nor
game data is checked into this repository. `LICENSES/` in the blob contains the Doomgeneric,
Freedoom, TinyCC, wasi-libc, and shellsim adapter notices. `PROVENANCE.json` records upstream
links, the optional Doom source revision, and the hashes of the bundled WAD and toolchain archives.
The display adapter is dual-licensed `Apache-2.0 OR GPL-2.0-or-later`; its GPL option permits
linking it into the Doom engine without relicensing the separate shellsim runtime.

The proven inputs were:

- [Doomgeneric](https://github.com/ozkl/doomgeneric) commit
  `dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284`, with
  `SHELLSIM_DOOM_SOURCE` pointing to its `doomgeneric/` source directory.
- [Freedoom 0.13.0](https://github.com/freedoom/freedoom/releases/tag/v0.13.0)
  release ZIP, containing `freedoom1.wad` with SHA-256
  `7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d`, with
  `SHELLSIM_FREEDOOM_ARCHIVE` pointing to the release ZIP. The builder reads its IWAD,
  `COPYING.txt`, and credits from the same archive.

Run the opt-in Python package probe with both variables set:

```sh
SHELLSIM_DOOM_SOURCE=/path/to/doomgeneric/doomgeneric \
SHELLSIM_FREEDOOM_ARCHIVE=/path/to/freedoom-0.13.0.zip \
uv run --group test --reinstall-package shellsim python -m pytest -q \
  tests/python_package/test_doom_package.py
```

To fetch the pinned external inputs and play from the repository root, run:

```sh
./examples/play-doom.sh
```

The launcher requires `curl`, `tar`, and `uv`. It downloads into a temporary
directory, verifies the IWAD hash while building a temporary `.shl`, then runs `shellsim run` against that
blob. The downloads and temporary package are removed after the player exits. To keep or publish
a package from inputs you obtained separately:

```sh
uv run python examples/build_doom_package.py \
  /path/to/doomgeneric/doomgeneric /path/to/freedoom-0.13.0.zip doom.shl \
  --doom-revision dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284 \
  --wad-sha256 7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d
uv run shellsim run ./doom.shl
uv run shellsim run https://example.com/doom.shl --sha256 <printed-package-digest>
```

GitHub release-asset URLs redirect to asset storage, while the default `.shl` HTTPS loader
rejects redirects. Download a GitHub asset first with `gh release download TAG --pattern '*.shl'`,
then run the local file with `uv run shellsim run ./doomgeneric-freedoom-0.13.0.shl --sha256 DIGEST`.
A non-redirecting HTTPS blob URL can be passed directly to `shellsim run`.

The package runner prints a loopback URL when Doom presents its first frame. Open it and click
the canvas to capture input. WASD or arrow keys move, J or Ctrl fires, K or Space uses, Shift
runs, and Esc opens the menu. The browser receives copied RGBA frames and sends bounded keys to
the host's generic `DisplayHost`; the guest has no network access. The host binds an ephemeral
`127.0.0.1` port and stops the guest when the Stop button is pressed. The external
[Freedoom release](https://github.com/freedoom/freedoom/releases/tag/v0.13.0) provides the WAD;
only the opt-in launcher downloads it. Inspect `LICENSES/` and `PROVENANCE.json` before
publishing; the source tree, toolchain, WAD, and adapter retain their respective license terms.

The builder selects Doomgeneric's console-error path instead of its optional Zenity `system()`
call. The adapter uses libc `clock_gettime` and `usleep`. The package runs at normal speed when
the CLI grants its real-time clock request; the probe uses virtual time. Audio and networking
remain outside this single-player demo.
