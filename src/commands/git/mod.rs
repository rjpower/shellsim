//! A small, deterministic Git porcelain subset backed entirely by the simulated VFS.
//!
//! The module is split by concern:
//!
//! * [`repo`] owns the on-disk layout below `.git`: objects, trees, commits, refs, and config.
//! * [`ignore`] matches `.gitignore` patterns.
//! * [`diff`] turns two byte buffers into a unified patch; [`compare`] renders tree differences
//!   and implements `git diff`.
//! * [`worktree`] implements the index and working-tree commands; [`history`] implements commits,
//!   history, refs, branch switching, and merges; [`config`] implements configuration and remotes.
//!
//! This file owns command registration, Git's global options, and dispatch. Anything outside the
//! supported subset fails visibly rather than approximating Git's behavior, and anything that
//! would need a network is refused and recorded as unsupported.

mod apply;
mod compare;
mod config;
mod conflict;
mod diff;
mod history;
mod ignore;
mod plumbing;
mod rebase;
mod repo;
mod stash;
mod worktree;

use std::collections::{BTreeMap, HashMap};

use crate::commands::{reg_costed, CommandContext, CommandSpec, Io, Trust};
use crate::vfs::{resolve_against, VfsError};

/// Version string reported by `git --version`. The suffix keeps the simulation identifiable
/// while leaving the usual `git version X.Y.Z` prefix parseable.
const VERSION: &str = "git version 2.45.0 (shellsim)";

const HELP: &str = "\
usage: git [-C DIRECTORY] [-c NAME=VALUE] COMMAND [ARGUMENTS]

Supported commands:
   add         stage working-tree contents
   apply       apply a unified diff to the working tree
   branch      list, create, rename, or delete branches
   cat-file    show an object's type, size, or contents
   check-ignore report which paths .gitignore excludes
   checkout    switch branches or restore files
   clean       remove untracked files
   commit      record staged changes
   config      read or write repository-local configuration
   describe    name a commit using the nearest tag
   diff        show changes between commits, the index, and the working tree
   grep        search tracked files
   hash-object compute a blob id
   init        create a repository
   log         show commit history
   ls-files    list files in the index and working tree
   ls-tree     list the contents of a tree
   merge       join two development histories
   merge-base  find a common ancestor
   mv          move or rename a tracked file
   remote      record remote names (the simulation has no network)
   reset       reset HEAD, the index, and optionally the working tree
   restore     restore working-tree or staged files
   rev-list    list commit ids reachable from a revision
   rev-parse   resolve revisions and repository paths
   rm          remove files from the index and working tree
   shortlog    summarize history by author
   show        show a commit or a file at a revision
   stash       save and restore uncommitted work
   status      show the working-tree status
   switch      switch branches
   tag         list, create, or delete tags
";

/// Subcommands that would need a network. They are refused rather than approximated.
/// Subcommands whose `-q` suppresses the report they would otherwise print.
const QUIET_COMMANDS: &[&str] = &[
    "add", "branch", "checkout", "clean", "commit", "merge", "mv", "reset", "restore", "rm",
    "stash", "switch", "tag",
];

/// Real Git subcommands this subset deliberately leaves out; anything else is simply not a
/// command, and is reported the way Git reports a typo.
const UNSUPPORTED_COMMANDS: &[&str] = &[
    "am",
    "bisect",
    "bundle",
    "filter-branch",
    "gc",
    "notes",
    "replace",
    "worktree",
];

const NETWORK_COMMANDS: &[&str] = &["clone", "fetch", "pull", "push", "submodule", "ls-remote"];

/// Settings `git init` records, matching the defaults a real repository starts with.
const DEFAULT_CONFIG: &[(&str, &str)] = &[
    ("core.bare", "false"),
    ("core.filemode", "true"),
    ("core.logallrefupdates", "true"),
    ("core.repositoryformatversion", "0"),
];

/// Command-line state that applies to every subcommand.
#[derive(Debug, Default)]
pub(crate) struct Globals {
    /// `-c NAME=VALUE` configuration overrides for this invocation only.
    pub overrides: BTreeMap<String, String>,
}

/// Whether this `git` invocation reads its message or object content from standard input.
///
/// Only `commit -F -` and `hash-object --stdin` do, so every other subcommand leaves a
/// redirected loop's input alone.
pub(crate) fn reads_standard_input(args: &[String]) -> bool {
    // `git apply` with no file operand reads the patch from standard input.
    if args.first().is_some_and(|subcommand| subcommand == "apply") {
        return !args[1..].iter().any(|argument| !argument.starts_with('-'));
    }
    let mut previous = "";
    for argument in args {
        if argument == "--stdin" || argument == "--file=-" {
            return true;
        }
        if argument == "-" && matches!(previous, "-F" | "--file") {
            return true;
        }
        previous = argument;
    }
    false
}

pub fn register(m: &mut HashMap<&'static str, CommandSpec>) {
    reg_costed(m, &["git"], Trust::Partial, 150, 16 * 1024, cmd_git);
}

/// An argument Git cannot make sense of: an option it does not have, or one used the wrong way.
///
/// Git exits 129 for these, which is how a caller tells a mistyped command from one that ran and
/// failed. Messages that already read as a synopsis are printed as they stand.
pub(crate) fn usage(io: &mut Io, message: &str) -> i32 {
    let line = if message.starts_with("usage:") {
        message.to_string()
    } else {
        format!("error: {message}")
    };
    io.err.extend_from_slice(format!("{line}\n").as_bytes());
    129
}

/// An operation that was understood but could not be carried out. Git exits 128 for these.
pub(crate) fn fatal(io: &mut Io, message: &str) -> i32 {
    io.err
        .extend_from_slice(format!("fatal: {message}\n").as_bytes());
    128
}

/// Report a repository write that could not be made, and the status to exit with.
///
/// These writes fail when the simulated disk fills. Returning a bare status for one leaves an
/// agent with a non-zero exit, nothing on standard error, and a repository it cannot reason
/// about, so every caller names what it was trying to write.
pub(crate) fn cannot_write(io: &mut Io, what: &str, error: &VfsError) -> i32 {
    fatal(io, &format!("unable to write {what}: {error}"))
}

/// The tree of a commit that may not be there: empty when there is none, fatal when there is one
/// and it cannot be read.
///
/// A root commit has no parent, and its change is measured against nothing. That is not the same
/// as a parent whose tree is missing, which must not be read as "every file was added".
pub(crate) fn parent_tree(
    ctx: &CommandContext<'_>,
    root: &str,
    parent: Option<&String>,
    io: &mut Io,
) -> Result<repo::Tree, i32> {
    match parent {
        Some(parent) => require_tree(ctx, root, parent, io),
        None => Ok(repo::Tree::new()),
    }
}

/// The staged tree, or a fatal error when the index cannot be read.
///
/// `repo::load_index` reads a *missing* index as nothing staged, which is what a fresh repository
/// has. What it cannot read is a *corrupt* one, and to a command that writes the index or the
/// working tree "nothing is staged" reads as "delete everything", so those ask for it this way.
pub(crate) fn require_index(
    ctx: &CommandContext<'_>,
    root: &str,
    io: &mut Io,
) -> Result<repo::Tree, i32> {
    repo::load_index(ctx, root).ok_or_else(|| fatal(io, "index file corrupt"))
}

/// One argument, as the command that asked for it sees it.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) enum Arg {
    /// An option, together with any value written against it: `--name=x` or `-nx`.
    Option {
        name: String,
        attached: Option<String>,
    },
    /// Anything else: a revision, a pathspec, a branch name.
    Operand(String),
}

/// One pass over a Git command's arguments.
///
/// Git spells an option's value four ways — `-n5`, `-n 5`, `--max-count=5`, `--max-count 5` — and
/// a command that writes its own loop tends to cover only the spellings someone needed, so which
/// ones work varies from command to command. `Flags` reads all four, splits short clusters, and
/// stops treating anything as an option after `--`, leaving each command a plain match on a name.
///
/// ```ignore
/// let mut flags = Flags::new(args).clustered("ne").valued("n");
/// while let Some(argument) = flags.next() {
///     match argument {
///         Arg::Option { name, attached } => match name.as_str() {
///             "-n" | "--max-count" => limit = flags.value(attached)?.parse().ok()?,
///             _ => return usage(io, &format!("unsupported log option: {name}")),
///         },
///         Arg::Operand(value) => operands.push(value),
///     }
/// }
/// ```
pub(crate) struct Flags {
    args: Vec<String>,
    at: usize,
    /// Set once `--` is seen: everything after it is a path, whatever it looks like.
    operands_only: bool,
    /// Short options that may carry their value written against them, as `-n5` does.
    valued: String,
}

impl Flags {
    pub(crate) fn new(args: &[String]) -> Self {
        Flags {
            args: args.to_vec(),
            at: 0,
            operands_only: false,
            valued: String::new(),
        }
    }

    /// Declare the short options that may be written as one cluster, as `-am` is.
    pub(crate) fn clustered(mut self, letters: &str) -> Self {
        self.args = expand_clusters(&self.args, letters);
        self
    }

    /// Declare the short options that may carry their value, so `-n5` reads as `-n` with `5`.
    pub(crate) fn valued(mut self, letters: &str) -> Self {
        self.valued = letters.to_string();
        self
    }

    /// Whether what is being read now came after `--`, and so is a path rather than a revision.
    pub(crate) fn separated(&self) -> bool {
        self.operands_only
    }

    /// The value written against an option, or else the argument that follows it.
    ///
    /// Only for options that require a value. An option whose value is optional, such as
    /// `--porcelain[=v2]`, must read `attached` directly: Git takes those only when they are
    /// written against the option, and what follows a bare one is the next operand.
    pub(crate) fn value(&mut self, attached: Option<String>) -> Option<String> {
        if attached.is_some() {
            return attached;
        }
        let value = self.args.get(self.at)?.clone();
        self.at += 1;
        Some(value)
    }

    /// The value written against an option, or the next argument when that is not an option.
    ///
    /// For options whose value may be left out, such as `git branch --contains`, which falls back
    /// to HEAD when no revision follows it.
    pub(crate) fn optional_value(&mut self, attached: Option<String>) -> Option<String> {
        if attached.is_some() {
            return attached;
        }
        let value = self.args.get(self.at)?;
        if value.starts_with('-') {
            return None;
        }
        let value = value.clone();
        self.at += 1;
        Some(value)
    }

    /// Everything left, as operands. Used by the commands that stop parsing at a given point.
    pub(crate) fn rest(&mut self) -> Vec<String> {
        let rest = self.args[self.at.min(self.args.len())..].to_vec();
        self.at = self.args.len();
        rest
    }
}

impl Iterator for Flags {
    type Item = Arg;

    fn next(&mut self) -> Option<Arg> {
        loop {
            let argument = self.args.get(self.at)?.clone();
            self.at += 1;
            if self.operands_only {
                return Some(Arg::Operand(argument));
            }
            if argument == "--" {
                self.operands_only = true;
                continue;
            }
            // A bare `-` is an operand: it is how `commit -F -` names standard input.
            if argument == "-" || !argument.starts_with('-') {
                return Some(Arg::Operand(argument));
            }
            if let Some(long) = argument.strip_prefix("--") {
                return Some(match long.split_once('=') {
                    Some((name, value)) => Arg::Option {
                        name: format!("--{name}"),
                        attached: Some(value.to_string()),
                    },
                    None => Arg::Option {
                        name: argument,
                        attached: None,
                    },
                });
            }
            let flags = &argument[1..];
            let first = flags.chars().next()?;
            let carries_value = flags.chars().count() > 1 && self.valued.contains(first);
            return Some(Arg::Option {
                name: if carries_value {
                    format!("-{first}")
                } else {
                    argument.clone()
                },
                attached: carries_value.then(|| flags[first.len_utf8()..].to_string()),
            });
        }
    }
}

/// Expand short-option clusters such as `-am` into `-a -m`.
///
/// Only clusters made entirely of `clusterable` letters are split, so spellings that carry an
/// attached value, such as `-U3` or `-n5`, reach their command untouched. Everything after `--`
/// is left alone because it is a pathspec.
pub(crate) fn expand_clusters(args: &[String], clusterable: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut operands_only = false;
    for argument in args {
        let splittable = !operands_only
            && argument.len() > 2
            && argument.starts_with('-')
            && !argument.starts_with("--")
            && argument[1..].chars().all(|flag| clusterable.contains(flag));
        if argument == "--" {
            operands_only = true;
        }
        if splittable {
            out.extend(argument[1..].chars().map(|flag| format!("-{flag}")));
        } else {
            out.push(argument.clone());
        }
    }
    out
}

/// Whether a short option or cluster of them carries `-q`, as in `git switch -qc`.
fn is_short_quiet(argument: &str) -> bool {
    let Some(flags) = argument.strip_prefix('-') else {
        return false;
    };
    !flags.is_empty()
        && !flags.starts_with('-')
        && flags.chars().all(|flag| flag.is_ascii_alphabetic())
        && flags.contains('q')
}

/// Convert a command-line pathspec into a repository-relative prefix.
///
/// An empty result means the repository root, which matches every path.
pub(crate) fn pathspec(cwd: &str, root: &str, value: &str) -> String {
    let absolute = resolve_against(cwd, value);
    match repo::relative_path(root, &absolute) {
        Some(relative) => relative,
        // Either the repository root itself, or a path outside it that will simply match nothing.
        None if repo::within(root, &absolute) => String::new(),
        None => value.to_string(),
    }
}

/// Whether an operand names something in the working tree or the index.
///
/// Git uses this to tell a mistyped revision from a pathspec; an operand that is neither is a
/// fatal ambiguous argument rather than a silently empty result.
/// The tree a commit records, or a fatal error naming the commit.
///
/// A missing tree reads as the empty tree, and to a command that writes the working tree an empty
/// tree means "delete every file", so the commands that write ask for a tree this way.
pub(crate) fn require_tree(
    ctx: &CommandContext<'_>,
    root: &str,
    commit: &str,
    io: &mut Io,
) -> Result<repo::Tree, i32> {
    repo::commit_tree(ctx, root, commit).ok_or_else(|| {
        io.err.extend_from_slice(
            format!("fatal: unable to read tree of commit {commit}\n").as_bytes(),
        );
        128
    })
}

pub(crate) fn names_a_path(ctx: &CommandContext<'_>, root: &str, value: &str) -> bool {
    let absolute = resolve_against(&ctx.cwd, value);
    if ctx.vfs.exists("/", &absolute) {
        return true;
    }
    let Some(relative) = repo::relative_path(root, &absolute) else {
        return true;
    };
    let prefix = format!("{relative}/");
    repo::load_index(ctx, root).is_some_and(|index| {
        index
            .keys()
            .any(|path| *path == relative || path.starts_with(&prefix))
    })
}

/// Git's diagnostic for an operand that is neither a revision nor a path.
pub(crate) fn ambiguous_argument(io: &mut Io, value: &str) -> i32 {
    io.err.extend_from_slice(
        format!(
            "fatal: ambiguous argument '{value}': unknown revision or path not in the working tree.\nUse '--' to separate paths from revisions, like this:\n'git <command> [<revision>...] -- [<file>...]'\n"
        )
        .as_bytes(),
    );
    128
}

pub(crate) fn repo_error(io: &mut Io) -> i32 {
    io.err.extend_from_slice(
        b"fatal: not a git repository (or any parent up to mount point /)\n\
              Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).\n",
    );
    128
}

fn cmd_git(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut globals = Globals::default();
    let mut directory: Option<String> = None;
    // The global options run out at the subcommand name, which is the first operand.
    let mut flags = Flags::new(args).valued("Cc");
    let mut rest = Vec::new();
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Operand(subcommand) => {
                rest.push(subcommand);
                rest.extend(flags.rest());
                break;
            }
            Arg::Option { name, attached } => (name, attached),
        };
        match name.as_str() {
            "-C" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "-C requires a directory");
                };
                // Repeated -C options compose, as they do in Git.
                let base = directory.as_deref().unwrap_or(&ctx.cwd);
                directory = Some(resolve_against(base, &value));
            }
            "-c" => {
                let Some((key, value)) = flags
                    .value(attached)
                    .as_deref()
                    .and_then(|pair| pair.split_once('='))
                    .map(|(key, value)| (key.to_ascii_lowercase(), value.to_string()))
                else {
                    return usage(io, "-c requires NAME=VALUE");
                };
                if !repo::valid_config_key(&key) {
                    return fatal(io, &format!("invalid config key: {key}"));
                }
                globals.overrides.insert(key, value);
            }
            "--no-pager"
            | "-P"
            | "--paginate"
            | "--no-replace-objects"
            | "--literal-pathspecs"
            | "--no-optional-locks" => {}
            "--version" => {
                io.out.extend_from_slice(format!("{VERSION}\n").as_bytes());
                return 0;
            }
            "--help" | "-h" => {
                io.out.extend_from_slice(HELP.as_bytes());
                return 0;
            }
            "--git-dir" | "--work-tree" => {
                return usage(
                    io,
                    &format!("{name} is unsupported; run git inside the tree"),
                );
            }
            _ => return usage(io, &format!("unsupported global option: {name}")),
        }
    }
    let args = &rest[..];
    if args.first().is_some_and(|value| value == "help") {
        io.out.extend_from_slice(HELP.as_bytes());
        return 0;
    }
    let Some(subcommand) = args.first().map(String::as_str) else {
        io.err.extend_from_slice(HELP.as_bytes());
        return 1;
    };
    let restore_cwd = match directory {
        Some(target) => {
            if !ctx.vfs.is_dir("/", &target) {
                io.err.extend_from_slice(
                    format!("fatal: cannot change to '{target}': No such file or directory\n")
                        .as_bytes(),
                );
                return 128;
            }
            let previous = ctx.cwd.clone();
            ctx.cwd = target;
            Some(previous)
        }
        None => None,
    };
    // `-q` silences the progress report these commands print; diagnostics still reach stderr.
    let silence = QUIET_COMMANDS.contains(&subcommand)
        && args[1..]
            .iter()
            .take_while(|argument| *argument != "--")
            .any(|argument| argument == "--quiet" || is_short_quiet(argument));
    let before = (io.out.len(), io.err.len());
    let status = dispatch(ctx, &globals, subcommand, &args[1..], io);
    if silence {
        io.out.truncate(before.0);
        // Progress notes such as `Switched to branch` go to standard error, as they do in Git;
        // a failure's diagnostics are kept.
        if status == 0 {
            io.err.truncate(before.1);
        }
    }
    if let Some(previous) = restore_cwd {
        ctx.cwd = previous;
    }
    status
}

fn dispatch(
    ctx: &mut CommandContext<'_>,
    globals: &Globals,
    subcommand: &str,
    args: &[String],
    io: &mut Io,
) -> i32 {
    match subcommand {
        "init" => git_init(ctx, args, io),
        "status" => worktree::git_status(ctx, args, io),
        "add" => worktree::git_add(ctx, args, io),
        "rm" => worktree::git_rm(ctx, args, io),
        "mv" => worktree::git_mv(ctx, args, io),
        "restore" => worktree::git_restore(ctx, args, io),
        "reset" => worktree::git_reset(ctx, args, io),
        "clean" => worktree::git_clean(ctx, args, io),
        "ls-files" => worktree::git_ls_files(ctx, args, io),
        "diff" => compare::git_diff(ctx, args, io),
        "commit" => history::git_commit(ctx, globals, args, io),
        "log" => history::git_log(ctx, args, io),
        "show" => history::git_show(ctx, args, io),
        "rev-parse" => history::git_rev_parse(ctx, args, io),
        "rev-list" => history::git_rev_list(ctx, args, io),
        "branch" => history::git_branch(ctx, args, io),
        "tag" => history::git_tag(ctx, globals, args, io),
        "switch" => history::git_switch(ctx, args, io),
        "checkout" => history::git_checkout(ctx, args, io),
        "blame" => plumbing::git_blame(ctx, args, io),
        "reflog" => plumbing::git_reflog(ctx, args, io),
        "rebase" => rebase::git_rebase(ctx, globals, args, io),
        "merge" => history::git_merge(ctx, globals, args, io),
        "cherry-pick" => history::git_replay(ctx, globals, false, args, io),
        "revert" => history::git_replay(ctx, globals, true, args, io),
        "config" => config::git_config(ctx, globals, args, io),
        "remote" => config::git_remote(ctx, globals, args, io),
        "stash" => stash::git_stash(ctx, args, io),
        "apply" => apply::git_apply(ctx, args, io),
        "grep" => plumbing::git_grep(ctx, args, io),
        "cat-file" => plumbing::git_cat_file(ctx, args, io),
        "hash-object" => plumbing::git_hash_object(ctx, args, io),
        "ls-tree" => plumbing::git_ls_tree(ctx, args, io),
        "check-ignore" => plumbing::git_check_ignore(ctx, args, io),
        "merge-base" => plumbing::git_merge_base(ctx, args, io),
        "describe" => plumbing::git_describe(ctx, args, io),
        "shortlog" => plumbing::git_shortlog(ctx, args, io),
        "show-ref" => plumbing::git_show_ref(ctx, args, io),
        "symbolic-ref" => plumbing::git_symbolic_ref(ctx, args, io),
        "for-each-ref" => plumbing::git_for_each_ref(ctx, args, io),
        other if NETWORK_COMMANDS.contains(&other) => {
            ctx.note_unsupported(&format!("git:{other}"));
            io.err.extend_from_slice(
                format!("fatal: git {other} needs network access, which the simulation does not provide\n")
                    .as_bytes(),
            );
            128
        }
        other if UNSUPPORTED_COMMANDS.contains(&other) => {
            ctx.note_unsupported(&format!("git:{other}"));
            usage(io, &format!("unsupported subcommand: {other}"))
        }
        other => {
            // An alias expands to another subcommand; one that names itself is not followed.
            if let Some(mut expansion) = config::alias(ctx, globals, other) {
                let name = expansion.remove(0);
                if name != other {
                    expansion.extend_from_slice(args);
                    return dispatch(ctx, globals, &name, &expansion, io);
                }
            }
            ctx.note_unsupported(&format!("git:{other}"));
            io.err.extend_from_slice(
                format!("git: '{other}' is not a git command. See 'git --help'.\n").as_bytes(),
            );
            1
        }
    }
}

fn git_init(ctx: &mut CommandContext<'_>, args: &[String], io: &mut Io) -> i32 {
    let mut branch = repo::DEFAULT_BRANCH.to_string();
    let mut quiet = false;
    let mut target = None;
    let mut flags = Flags::new(args).clustered("q").valued("b");
    while let Some(argument) = flags.next() {
        let (name, attached) = match argument {
            Arg::Option { name, attached } => (name, attached),
            Arg::Operand(value) if target.is_none() => {
                target = Some(value);
                continue;
            }
            Arg::Operand(_) => return usage(io, "usage: git init [-b BRANCH] [DIRECTORY]"),
        };
        match name.as_str() {
            "--bare" => return usage(io, "--bare is unsupported"),
            "-q" | "--quiet" => quiet = true,
            "-b" | "--initial-branch" => {
                let Some(value) = flags.value(attached) else {
                    return usage(io, "-b requires a branch name");
                };
                branch = value;
            }
            _ => return usage(io, &format!("unsupported init option: {name}")),
        }
    }
    let target = target
        .map(|path| resolve_against(&ctx.cwd, &path))
        .unwrap_or_else(|| ctx.cwd.clone());
    if !ctx.vfs.is_dir("/", &target) {
        if let Err(error) = ctx.vfs.mkdir_all("/", &target) {
            io.err
                .extend_from_slice(format!("git init: {error}\n").as_bytes());
            return 1;
        }
    }
    let git = repo::path_join(&target, repo::GIT_DIR);
    let reinitialized = ctx.vfs.is_dir("/", &git);
    for directory in [
        git.clone(),
        repo::path_join(&git, "refs/heads"),
        repo::path_join(&git, "refs/tags"),
        repo::path_join(&git, "commits"),
        repo::path_join(&git, "objects"),
    ] {
        if let Err(error) = ctx.vfs.mkdir_all("/", &directory) {
            io.err
                .extend_from_slice(format!("git init: {error}\n").as_bytes());
            return 1;
        }
    }
    if !reinitialized {
        for (path, contents) in [
            (
                repo::git_path(&target, repo::HEAD),
                format!("ref: refs/heads/{branch}\n").into_bytes(),
            ),
            (repo::git_path(&target, repo::INDEX), Vec::new()),
            (
                repo::git_path(&target, repo::CONFIG),
                repo::serialize_config(
                    &DEFAULT_CONFIG
                        .iter()
                        .map(|(key, value)| ((*key).to_string(), vec![(*value).to_string()]))
                        .collect(),
                ),
            ),
        ] {
            if let Err(error) = repo::write_vfs(ctx, &path, &contents) {
                io.err
                    .extend_from_slice(format!("git init: {error}\n").as_bytes());
                return 1;
            }
        }
    }
    if !quiet {
        let action = if reinitialized {
            "Reinitialized existing"
        } else {
            "Initialized empty"
        };
        io.out
            .extend_from_slice(format!("{action} Git repository in {git}/\n").as_bytes());
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{Arg, Flags};

    fn parse(args: &[&str]) -> Vec<Arg> {
        let args: Vec<String> = args.iter().map(|value| (*value).to_string()).collect();
        Flags::new(&args).clustered("am").valued("n").collect()
    }

    fn option(name: &str, attached: Option<&str>) -> Arg {
        Arg::Option {
            name: name.to_string(),
            attached: attached.map(str::to_string),
        }
    }

    #[test]
    fn all_four_spellings_of_a_value_read_the_same() {
        let spellings = [
            vec!["-n5"],
            vec!["-n", "5"],
            vec!["--max-count=5"],
            vec!["--max-count", "5"],
        ];
        for spelling in spellings {
            let args: Vec<String> = spelling.iter().map(|value| (*value).to_string()).collect();
            let mut flags = Flags::new(&args).valued("n");
            let Some(Arg::Option { attached, .. }) = flags.next() else {
                panic!("{spelling:?} did not read as an option");
            };
            assert_eq!(flags.value(attached).as_deref(), Some("5"), "{spelling:?}");
        }
    }

    #[test]
    fn a_cluster_splits_into_its_letters() {
        assert_eq!(
            parse(&["-am"]),
            vec![option("-a", None), option("-m", None)]
        );
    }

    #[test]
    fn everything_after_a_separator_is_an_operand() {
        assert_eq!(
            parse(&["-a", "--", "-a", "--max-count=5"]),
            vec![
                option("-a", None),
                Arg::Operand("-a".to_string()),
                Arg::Operand("--max-count=5".to_string()),
            ]
        );
    }

    #[test]
    fn a_bare_dash_is_an_operand() {
        assert_eq!(parse(&["-"]), vec![Arg::Operand("-".to_string())]);
    }

    #[test]
    fn an_optional_value_does_not_swallow_the_next_option() {
        let args: Vec<String> = ["--contains", "--merged"]
            .iter()
            .map(|value| (*value).to_string())
            .collect();
        let mut flags = Flags::new(&args);
        let Some(Arg::Option { attached, .. }) = flags.next() else {
            panic!("--contains did not read as an option");
        };
        assert_eq!(flags.optional_value(attached), None);
        assert_eq!(flags.next(), Some(option("--merged", None)));
    }
}
