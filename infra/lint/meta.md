# Whole-change lane

Apply only the rules in this file. This lane may inspect beyond changed hunks, but it reports only
shapes that require seeing the change as a whole. It runs only for changes above the runner's size
threshold. Keep every claim concrete and suppress below 0.85 confidence.

### `ml-there-must-be-a-better-way` — the added surface has a smaller concrete shape

Report only when one named existing helper, struct, enum, trait, protocol, or dispatch table would
remove several additions without weakening a boundary. Cite the construct and the redundant
surface; do not request an abstract "cleaner design."

### `ml-data-clump-threaded` — a group or knob is forwarded through several layers

Report the same three-value group added to at least two signatures, or one new value forwarded
unchanged through at least three functions when an existing context object already spans the
path. Numeric coordinates and values consumed at each layer are allowed.

### `ml-shotgun-surgery` — one fact requires logic at many sites

Report the same conceptual guard, branch, or registration added at four or more sites when a
specific central seam would reduce it to zero call-site logic. Required-field propagation and
renames are allowed.

### `ml-echo-across-files` — the change introduces cross-file duplication

Report structurally equivalent blocks of at least eight logical lines, or the same nontrivial
literal set, added in multiple files when a layer-correct shared home already exists.

### `ml-reinvents-existing-helper` — new code duplicates a verified repository helper

Report only after reading the existing symbol and confirming equivalent behavior and legal
dependency direction. Name the existing path and symbol.

### `ml-asymmetric-pair` — a lifecycle operation lacks its counterpart

Report recognized pairs such as open/close, acquire/release, mount/unmount, register/unregister,
or add/remove only after searching the whole tree. Ask for confirmation when dispatch could hide
the counterpart.

### `ml-orphaned-by-change` — the change strands an unchanged old path

Report as a confirmation question only when the diff introduces a replacement, removes the last
static caller, and a whole-tree search finds no registry, export, string reference, or command
dispatch use. Published compatibility surfaces may remain intentionally.
