Doomgeneric is compiled once by the host SDK and installed as `/usr/bin/doom`.
It uses shellsim's virtual display, input and clock adapter from PR92; sound is disabled.
Its runtime dependency installs Freedoom 0.13.0: run `doom -iwad /usr/share/games/doom/freedoom1.wad`.
The pinned source, GPL notice and dated adapter patch remain part of the build inputs.
Use `uv run examples/play_doom.py --catalog <catalog>` with the parent delivery bridge;
the browser viewer and controls are unchanged from PR92.
