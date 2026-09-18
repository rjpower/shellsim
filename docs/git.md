# Git in shellsim

Shellsim implements a Git porcelain subset directly against its in-memory filesystem. Repository
state lives below `.git`; no host Git process, host filesystem, or network is involved. The goal
is that an agent dropped into a working tree can orient itself, stage work, commit, inspect
history, branch, and merge without noticing it is not talking to Git.

## What is supported

```text
init  status  add  rm  mv  restore  reset  clean  ls-files
commit  log  show  diff  rev-parse  rev-list  branch  tag  switch  checkout  merge
config  remote  stash  grep  cat-file  hash-object  ls-tree  check-ignore  merge-base
describe  shortlog
```

Global options `-C DIRECTORY`, `-c NAME=VALUE`, `--no-pager`, `--version`, and `git help` work
before any subcommand. `git help` lists the supported set.

Output follows real Git byte for byte wherever an agent is likely to parse it: `git status` long
and `--porcelain` formats including untracked-directory collapsing and exact rename detection,
unified diffs with hunk headers, function context, `\ No newline at end of file`, and index lines
carrying real Git blob ids, `git log` author and date headers, `--format` placeholders, and the
`N files changed, N insertions(+)` summaries. `.gitignore` is honored by `status`, `add`, `clean`,
`ls-files --others --exclude-standard`, and `check-ignore`, including negation, anchoring,
directory-only patterns, `**`, and nested pattern files.

Revisions accept `HEAD`, `@`, branch and tag names, full and abbreviated commit ids, the `~N` and
`^N` ancestry suffixes, and the `a..b` and `a...b` ranges used by `log` and `diff`.

Configuration is stored in Git's INI format in `.git/config`, so a block appended by hand or by
another tool reads back correctly. There are two scopes: the repository and a per-user
`~/.gitconfig` inside the virtual filesystem. Commit identity comes from `GIT_AUTHOR_NAME` and
`GIT_AUTHOR_EMAIL`, then `user.name` and `user.email`, then a `shellsim <shellsim@localhost>`
default.

## What is not supported

Anything needing a network is refused and recorded as unsupported: `clone`, `fetch`, `pull`,
`push`, `ls-remote`, and `submodule`. `git remote` records remote names but never contacts one.

Content conflicts are refused rather than written as markers. `git merge` fast-forwards when it
can and otherwise takes whichever side changed each path; when both sides changed one file it
reports the conflicting paths, leaves the working tree untouched, and exits nonzero. There is no
`rebase`, `cherry-pick`, `revert`, `reflog`, `bisect`, `blame`, or `worktree`, and no interactive
mode for any command.

File modes, symbolic links, submodules, and rename detection based on similarity are outside the
model: every tracked path is a regular file, and renames are recognized only when the content is
byte-identical. Annotated tags store a tagger and message but are not separate objects.

## Storage layout

The format below `.git` is private to the simulator and deliberately simple. A repository created
by real Git is not readable, and a repository created here is not meant to be handed to real Git.

```text
.git/HEAD              symbolic or direct reference to the current commit
.git/index             the staged tree
.git/refs/heads/*      branches
.git/refs/tags/*       tags
.git/tags/*.annotation tagger and message for annotated tags
.git/objects/<sha1>    blob content, addressed by Git's blob hash
.git/commits/<id>.tree     a commit's tree: path to blob hash
.git/commits/<id>.commit   a commit's parents, author, timestamp, and message
.git/config            repository configuration in Git's INI format
.git/stash-list        saved stash entries, newest first
```

Trees are flat path maps rather than nested tree objects, and commits are stored uncompressed.
Blob ids match real Git's (`sha1("blob <len>\0" + content)`), so `git hash-object` and the `index`
lines in diffs agree with a real repository; commit ids do not, because commits do not serialize
real tree objects.

## Limits

Every command meters the work it can be asked to do. Working-tree snapshots are bounded in file
count and reserve memory before hashing, diffs trim their common prefix and suffix before running
a quadratic matcher and fall back to a whole-file replacement past a size threshold, `git grep`
bounds the number of files it searches, and `.gitignore` bounds the number of pattern files and
patterns. Exceeding a limit stops the command with the environment's resource status rather than
doing unbounded work.
