# Git in shellsim

Shellsim implements a Git porcelain subset directly against its in-memory filesystem. Repository
state lives below `.git`; no host Git process, host filesystem, or network is involved. The goal
is that an agent dropped into a working tree can orient itself, stage work, commit, inspect
history, branch, and merge without noticing it is not talking to Git.

## What is supported

```text
init  status  add  rm  mv  restore  reset  clean  ls-files
commit  log  show  diff  apply  rev-parse  rev-list  branch  tag  switch  checkout  merge
cherry-pick  revert  rebase  config  remote  stash  grep  cat-file  hash-object  ls-tree
check-ignore  merge-base  describe  shortlog  show-ref  symbolic-ref  for-each-ref  blame  reflog
```

Every move of HEAD is recorded, so `git reflog` lists where it has been and `HEAD@{N}` names the
commit it was on N moves ago. That is what makes a mistaken `git reset --hard` recoverable. Only
HEAD is logged, not individual branches, and the last thousand moves are kept. A reset, merge or
rebase also writes `ORIG_HEAD`, so `git reset --hard ORIG_HEAD` undoes one.

`git blame` walks first parents, carrying each line back until the version that introduced it,
and marks lines the working tree has not committed with a zero hash. It follows a file back
through a commit that only renamed it; a rename that also edited the file stops the walk there.
`git log --graph` draws the branch rail, with connectors on lines of their own rather than folded
into the commit header the way Git does it. The rail is right for one line of development and the
merges that close into it; `--all` over branches that never merged can put a collapse in the
wrong column.

`git apply` holds a patch to the anchors Git holds it to: a hunk with no trailing context has to
reach the end of the file, and one that starts at the first line has to start there. A hunk that
does not fit where it says it goes is looked for elsewhere before the patch is refused.

Aliases recorded in configuration are expanded, so `git config alias.st "status --short"`
makes `git st` work. A `!shell command` alias is not, because it would need host execution.

Global options `-C DIRECTORY`, `-c NAME=VALUE`, `--no-pager`, `--version`, and `git help` work
before any subcommand. `git help` lists the supported set.

Output follows real Git byte for byte wherever an agent is likely to parse it: `git status` long
and `--porcelain` formats including untracked-directory collapsing and exact rename detection,
unified diffs with hunk headers, function context, `\ No newline at end of file`, and index lines
carrying real Git blob ids, `git status --porcelain=v2` records, `git log` author and date headers, `--format`
placeholders, and the
`N files changed, N insertions(+)` summaries. `.gitignore` and `.git/info/exclude` are honored by
`status`, `add`, `clean`, `ls-files --others --exclude-standard`, and `check-ignore`, including
negation, anchoring, directory-only patterns, character classes, `**`, and nested pattern files.
Naming an ignored file outright to `git add` is an error, as it is in Git.

Options are read the way Git reads them, whichever of its four spellings a caller reaches for:
`-n5`, `-n 5`, `--max-count=5` and `--max-count 5` are one option, short options cluster as `-am`
does, and nothing after `--` is treated as an option however it is written.

Pathspecs accept globs, and their wildcards cross directory separators the way Git's do, so
`git status -- '*.py'` reports a change to `src/main.py`. `git grep` searches only below the
working directory, reports paths relative to it, and accepts a revision to search instead of the
working tree.

`git log` filters with `--grep`, `--author`, `-S`, `-G`, `--since`, and `--until`; dates are read
as `YYYY-MM-DD[ HH:MM:SS]`, a bare epoch second count, or Git's `N units ago`, and a form that is
not one of those is refused rather than guessed at. `GIT_AUTHOR_DATE` sets a new commit's author
date, as it does in Git.

`git apply` handles `--index`, `--cached`, `--check`, `-R`, `-p<n>`, `--stat`, and `--numstat`,
and reads the patch from a file or standard input.

Revisions accept `HEAD`, `@`, branch and tag names, full and abbreviated commit ids, the `~N` and
`^N` ancestry suffixes, the `a..b` and `a...b` ranges used by `log` and `diff`, and the
`<rev>:<path>`, `<rev>^{tree}`, and `<rev>^{commit}` forms `rev-parse` and `cat-file` resolve.
`git log` walks every parent of a merge unless `--first-parent` is given.

Moving between commits preserves uncommitted work: `checkout`, `switch`, `merge`, and
`stash pop` rewrite only the paths the move changes, and refuse outright when a path they must
rewrite holds uncommitted changes. `git restore` and `git reset --hard` still overwrite, which is
what they are for.

Configuration is stored in Git's INI format in `.git/config`, so a block appended by hand or by
another tool reads back correctly. There are two scopes: the repository and a per-user
`~/.gitconfig` inside the virtual filesystem. Commit identity comes from `GIT_AUTHOR_NAME` and
`GIT_AUTHOR_EMAIL`, then `user.name` and `user.email`, then a `shellsim <shellsim@localhost>`
default.

## Conflicts

`git merge` fast-forwards when it can, and otherwise merges each file three ways against the
merge base. A file both sides changed in the same region is written with `<<<<<<<`, `=======`,
and `>>>>>>>` markers, and the three sides are recorded so the rest of the workflow behaves:
`git status` reports the path as `UU`, `AA`, `UD`, or `DU` and lists it under "Unmerged paths",
`git ls-files -u` prints the stages, `git commit` refuses until the path is staged, `git add` or
`git rm` marks it resolved, and the finished commit carries both parents. `git merge --abort`
restores the pre-merge state without touching untracked files, and `git merge --continue` is the
same as committing.

`git checkout --ours`/`--theirs` and `git restore --ours`/`--theirs` put one recorded side back in
the working tree, leaving the path unmerged until it is staged. While a path is unmerged `git
diff` compares the working file against the side that was ours going into the merge, so the
markers show up in a review and `--diff-filter=U` names the path.

`git cherry-pick` and `git revert` run the same three-way merge with different corners: a
cherry-pick uses the commit's parent as the base and the commit as the incoming side, and a
revert swaps those two. Both stop at a conflict, both take `--continue`, `--skip` and `--abort`,
and both want a settled index first because the commit they make would otherwise fold in whatever
was already staged. `-n` makes no commit, so it goes ahead and leaves other staged work alone. A
cherry-pick keeps the original author; a revert does not. `-m PARENT` names which parent of a
merge commit the change is measured against, which is what lets a merge be replayed or undone.

Several commits replay as a list. What is left of it is recorded in `.git/SEQUENCER`, so a
conflict partway through does not lose the rest: `--continue` and `--skip` carry on through the
remaining commits, and `--abort` undoes the whole run rather than leaving it half applied.

`git stash pop` and `git stash apply` merge the entry back the same way, against the commit it was
taken from, so work committed or edited in the meantime survives. A plain reapplication restores
the working tree only, as Git's does, except that a file nothing tracks yet is staged, because
there is no way to leave it unstaged. An entry that conflicts stays on the list. `git stash push`
refuses while a path is unmerged, since one saved tree cannot hold three sides.

`git rebase [--onto NEWBASE] UPSTREAM [BRANCH]` replays the commits the branch has and the
upstream does not, one at a time, as a run of cherry-picks; naming BRANCH checks it out first. It
refuses to start on a working tree or index that has moved away from HEAD, because the replay
would overwrite that work. HEAD is detached for the replay, as Git detaches it, so the branch
keeps naming its old tip until the rebase succeeds, and `git switch` is refused until it ends. It
stops at the first conflict and takes `--continue`, `--skip`, and `--abort`; `--abort` puts the
branch back exactly where it was. A commit whose change is already in the new base is dropped.
Interactive rebase needs an editor and is refused.

While a merge, cherry-pick, revert or rebase is unfinished, `git status` names the operation and
the `--continue`, `--skip`, and `--abort` forms that belong to it.

A file one side renamed without changing it is followed, so the other side's edit lands under
the new name instead of reading as a deletion of one file and an addition of another. A file that
was renamed *and* edited is not followed, and two sides that renamed the same file differently
are left as two files rather than reported as a rename/rename conflict.

Binary files and a path one side deleted while the other changed it are recorded as conflicts
without markers, leaving the surviving content in the working tree, as Git does.

`checkout`, `switch`, `merge` and `rebase` refuse to write over an untracked file, because
nothing has recorded it and the content could not be got back. A move between commits keeps what
was staged rather than replacing the index with the tree it moved to, and no move is allowed
while a merge, replay or rebase is unfinished.

## What is not supported

Anything needing a network is refused and recorded as unsupported: `clone`, `fetch`, `pull`,
`push`, `ls-remote`, and `submodule`. `git remote` records remote names but never contacts one.

There is no `bisect` or `worktree`, and no interactive mode for any
command. Those report `unsupported subcommand` and exit 129, which is the status Git uses for an
argument it cannot make sense of; a name that is not a Git subcommand at all is reported the way
Git reports a typo and exits 1. An operation that was understood but could not be carried out —
an unresolvable revision, a pathspec matching nothing — prints `fatal:` and exits 128, again as
Git does.

A repository that cannot be read is refused rather than guessed at. An absent index file is
nothing staged, as it is in Git, but an index that is there and unreadable stops every command
that would write the index or the working tree, because an empty index reads as "delete
everything". The same holds for a commit whose tree is missing: the commands that write ask for
it in a form that fails. A write that cannot be made — the simulated disk has a size limit — says
what it could not write rather than exiting quietly.

Submodules and rename detection based on similarity are outside the model: `status` and `diff`
recognize a rename only when the content is byte-identical. Modes are tracked as Git tracks them:
`chmod +x` on a tracked file shows up in `status` and as `old mode`/`new mode` in `diff`, a
checkout restores it so a committed script stays runnable, and a symbolic link is recorded under
mode `120000` as a blob holding its target. Annotated tags store a tagger and message but are not
separate objects.

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
.git/commits/<id>.tree     a commit's tree: path to blob hash, and mode when not plain
.git/commits/<id>.commit   a commit's parents, author, timestamp, and message
.git/config            repository configuration in Git's INI format
.git/stash-list        saved stash entries, newest first
.git/logs/HEAD         every move of HEAD, oldest first, as `before<TAB>after<TAB>action`
.git/REBASE_STATE      the branch, its original tip, what it replays onto, and what is left
.git/ORIG_HEAD         where HEAD was before the last reset, merge, or rebase
.git/MERGE_HEAD        the commit an unfinished merge is bringing in
.git/CHERRY_PICK_HEAD  the commit an unfinished cherry-pick is replaying
.git/REVERT_HEAD       the commit an unfinished revert is undoing
.git/MERGE_MSG         the message that operation will commit
.git/MERGE_STAGES      the three sides of each unmerged path, as `stage<TAB>hash<TAB>path`
.git/SEQUENCER         the commits a multi-commit cherry-pick or revert has left to replay
```

Trees are flat path maps rather than nested tree objects, and commits are stored uncompressed.
Blob ids match real Git's (`sha1("blob <len>\0" + content)`), so `git hash-object` and the `index`
lines in diffs agree with a real repository; commit and tree ids do not, because neither
serializes a real tree object. Tree ids are still self-consistent: `ls-tree`, `cat-file`, and
`rev-parse <rev>^{tree}` all report the same value for the same tree.

## Limits

Every command meters the work it can be asked to do. Working-tree snapshots are bounded in file
count and reserve memory before hashing, diffs trim their common prefix and suffix before running
a quadratic matcher and fall back to a whole-file replacement past a size threshold, `git grep`
bounds the number of files it searches, and `.gitignore` bounds the number of pattern files and
patterns. Exceeding a limit stops the command with the environment's resource status rather than
doing unbounded work.
