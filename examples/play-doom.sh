#!/usr/bin/env bash
# Fetch the pinned external demo inputs into a temporary directory, then run the browser player.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workdir=$(mktemp -d "${TMPDIR:-/tmp}/shellsim-doom.XXXXXX")
trap 'rm -rf -- "$workdir"' EXIT

doom_commit=dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284
doom_archive="$workdir/doomgeneric.tar.gz"
freedoom_archive="$workdir/freedoom.zip"
wad="$workdir/freedoom1.wad"

printf 'Fetching Doomgeneric at %s...\n' "$doom_commit"
curl --fail --location --retry 3 --silent --show-error \
  --output "$doom_archive" \
  "https://codeload.github.com/ozkl/doomgeneric/tar.gz/$doom_commit"
tar -xzf "$doom_archive" -C "$workdir"
source_dir="$workdir/doomgeneric-$doom_commit/doomgeneric"
test -f "$source_dir/Makefile.soso"

printf 'Fetching Freedoom 0.13.0...\n'
curl --fail --location --retry 3 --silent --show-error \
  --output "$freedoom_archive" \
  'https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip'
unzip -p "$freedoom_archive" 'freedoom-0.13.0/freedoom1.wad' > "$wad"
printf '%s  %s\n' \
  '7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d' "$wad" \
  | shasum -a 256 -c

printf 'Starting the shellsim browser player...\n'
cargo run --locked --manifest-path "$repo_root/Cargo.toml" --example doom_player -- \
  "$source_dir" "$wad"
