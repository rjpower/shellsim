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
mod diff;
mod history;
mod ignore;
mod plumbing;
mod repo;
mod stash;
mod worktree;

use std::collections::{BTreeMap, HashMap};

use crate::commands::{reg_costed, CommandContext, CommandSpec, Io, Trust};
use crate::vfs::resolve_against;

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
    "blame",
    "bundle",
    "cherry-pick",
    "filter-branch",
    "gc",
    "notes",
    "rebase",
    "reflog",
    "replace",
    "revert",
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

pub(crate) fn usage(io: &mut Io, message: &str) -> i32 {
    io.err
        .extend_from_slice(format!("git: {message}\n").as_bytes());
    2
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
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].as_str();
        match argument {
            "-C" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "-C requires a directory");
                };
                // Repeated -C options compose, as they do in Git.
                let base = directory.as_deref().unwrap_or(&ctx.cwd);
                directory = Some(resolve_against(base, value));
            }
            "-c" => {
                index += 1;
                let Some((key, value)) = args.get(index).and_then(|pair| pair.split_once('='))
                else {
                    return usage(io, "-c requires NAME=VALUE");
                };
                let key = key.to_ascii_lowercase();
                if !repo::valid_config_key(&key) {
                    return usage(io, &format!("invalid config key: {key}"));
                }
                globals.overrides.insert(key, value.to_string());
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
            "--help" | "-h" | "help" => {
                io.out.extend_from_slice(HELP.as_bytes());
                return 0;
            }
            value if value.starts_with("--git-dir") || value.starts_with("--work-tree") => {
                return usage(
                    io,
                    &format!("{value} is unsupported; run git inside the tree"),
                );
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported global option: {value}"));
            }
            _ => break,
        }
        index += 1;
    }
    let args = &args[index..];
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
            .any(|argument| argument == "-q" || argument == "--quiet");
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
        "merge" => history::git_merge(ctx, globals, args, io),
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
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--bare" => return usage(io, "--bare is unsupported"),
            "-q" | "--quiet" => quiet = true,
            "-b" | "--initial-branch" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return usage(io, "-b requires a branch name");
                };
                branch = value.clone();
            }
            value if value.starts_with("--initial-branch=") => {
                branch = value["--initial-branch=".len()..].to_string();
            }
            value if value.starts_with('-') => {
                return usage(io, &format!("unsupported init option: {value}"))
            }
            value if target.is_none() => target = Some(value.to_string()),
            _ => return usage(io, "usage: git init [-b BRANCH] [DIRECTORY]"),
        }
        index += 1;
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
