//! Integration coverage for the deterministic, VFS-only Git porcelain subset.
//!
//! Tests exercise Git through the normal shell dispatcher and inspect only simulated files. They
//! cover the index/blob boundary because cached content must not change when the working file does.

use shellsim::Environment;

fn run(env: &mut Environment, command: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = env.run_script_capture(command);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn stages_commits_and_reports_working_tree_changes() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/note.txt", b"first\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(run(&mut env, "git status --short").1, "?? note.txt\n");
    assert_eq!(run(&mut env, "git add note.txt").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "A  note.txt\n");
    assert_eq!(run(&mut env, "git commit -m initial").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "");

    env.vfs
        .put_file("/note.txt", b"second\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git status --porcelain").1, " M note.txt\n");
    let diff = run(&mut env, "git diff");
    assert_eq!(diff.0, 0, "{}", diff.2);
    assert!(diff.1.contains("-first\n+second\n"), "{}", diff.1);
}

#[test]
fn cached_diff_reads_staged_blobs_not_the_working_file() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/data", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add data; git commit -m one").0, 0);

    env.vfs.put_file("/data", b"two\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add data").0, 0);
    env.vfs
        .put_file("/data", b"three\n".to_vec(), 0o644)
        .unwrap();

    let cached = run(&mut env, "git diff --cached");
    assert_eq!(cached.0, 0, "{}", cached.2);
    assert!(cached.1.contains("-one\n+two\n"), "{}", cached.1);
    assert!(!cached.1.contains("three"), "{}", cached.1);

    let working = run(&mut env, "git diff");
    assert!(working.1.contains("-two\n+three\n"), "{}", working.1);
}

#[test]
fn resolves_head_and_rejects_unsupported_operations() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init project").0, 0);
    env.vfs
        .put_file("/project/file", b"value".to_vec(), 0o644)
        .unwrap();
    let committed = run(&mut env, "cd project; git add file; git commit -m snapshot");
    assert_eq!(committed.0, 0, "{}", committed.2);

    let head = run(&mut env, "git rev-parse HEAD");
    assert_eq!(head.0, 0, "{}", head.2);
    assert_eq!(head.1.trim().len(), 40);
    assert!(head.1.trim().bytes().all(|byte| byte.is_ascii_hexdigit()));

    let unsupported = run(&mut env, "git clone nowhere");
    assert_eq!(unsupported.0, 128);
    assert!(
        unsupported.2.contains("needs network access"),
        "{}",
        unsupported.2
    );

    let unknown = run(&mut env, "git bisect start");
    assert_eq!(unknown.0, 2);
    assert!(
        unknown.2.contains("unsupported subcommand"),
        "{}",
        unknown.2
    );
}

#[test]
fn branches_switch_worktrees_and_preserve_independent_history() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/file", b"main\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m initial").0, 0);
    let initial = run(&mut env, "git rev-parse HEAD").1.trim().to_string();

    assert_eq!(run(&mut env, "git switch -c feature").0, 0);
    env.vfs
        .put_file("/file", b"feature\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m feature").0, 0);
    let feature = run(&mut env, "git rev-parse HEAD").1.trim().to_string();
    assert_ne!(initial, feature);

    assert_eq!(run(&mut env, "git switch main").0, 0);
    assert_eq!(env.vfs.read("/", "/file").unwrap(), b"main\n");
    assert_eq!(run(&mut env, "git rev-parse --abbrev-ref HEAD").1, "main\n");
    assert_eq!(run(&mut env, "git branch --show-current").1, "main\n");
    assert_eq!(run(&mut env, "git branch").1, "  feature\n* main\n");
    assert_eq!(
        run(&mut env, "git log --oneline").1,
        format!("{} initial\n", &initial[..7])
    );
    let shown = run(&mut env, "git show feature");
    assert_eq!(shown.0, 0, "{}", shown.2);
    assert!(shown.1.contains("feature"), "{}", shown.1);
    assert!(shown.1.contains("-main\n+feature\n"), "{}", shown.1);

    env.vfs
        .put_file("/file", b"changed\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git diff --name-only").1, "file\n");
    assert!(run(&mut env, "git status")
        .1
        .starts_with("On branch main\n"));
}

#[test]
fn restore_and_reset_move_index_worktree_and_head_coherently() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/file", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m one").0, 0);
    let first = run(&mut env, "git rev-parse HEAD").1.trim().to_string();

    env.vfs.put_file("/file", b"two\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git restore file").0, 0);
    assert_eq!(env.vfs.read("/", "/file").unwrap(), b"one\n");

    env.vfs.put_file("/file", b"two\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m two").0, 0);
    env.vfs
        .put_file("/staged", b"remove me\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/untracked", b"keep me\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add staged").0, 0);
    assert_eq!(run(&mut env, &format!("git reset --hard {first}")).0, 0);
    assert_eq!(run(&mut env, "git rev-parse HEAD").1.trim(), first);
    assert_eq!(env.vfs.read("/", "/file").unwrap(), b"one\n");
    assert!(!env.vfs.exists("/", "/staged"));
    assert_eq!(env.vfs.read("/", "/untracked").unwrap(), b"keep me\n");
    assert_eq!(run(&mut env, "git status --short").1, "?? untracked\n");
    assert_eq!(run(&mut env, "git log -1 --oneline").1.lines().count(), 1);
}

#[test]
fn local_config_and_tracked_file_queries_support_agent_setup() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init project").0, 0);
    assert_eq!(
        run(
            &mut env,
            "cd /project; git config user.name Agent; git config user.email agent@example.test; git config --get user.name; git config --list",
        ),
        (
            0,
            concat!(
                "Agent\n",
                "core.bare=false\n",
                "core.filemode=true\n",
                "core.logallrefupdates=true\n",
                "core.repositoryformatversion=0\n",
                "user.email=agent@example.test\n",
                "user.name=Agent\n",
            )
            .into(),
            String::new()
        )
    );
    env.vfs
        .put_file("/project/top.txt", b"top\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/project/src/lib.rs", b"lib\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "cd /project; git add .").0, 0);
    assert_eq!(
        run(&mut env, "cd /project; git ls-files").1,
        "src/lib.rs\ntop.txt\n"
    );
    assert_eq!(run(&mut env, "cd /project/src; git ls-files").1, "lib.rs\n");
    assert_eq!(
        run(&mut env, "cd /project/src; git rev-parse --show-prefix").1,
        "src/\n"
    );
    // Git prints a relative git directory at the repository root and an absolute one below it.
    assert_eq!(
        run(&mut env, "cd /project; git rev-parse --git-dir").1,
        ".git\n"
    );
    assert_eq!(
        run(&mut env, "cd /project/src; git rev-parse --git-dir").1,
        "/project/.git\n"
    );
}

#[test]
fn diff_check_reports_new_trailing_whitespace() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/file", b"clean\nend\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m clean").0, 0);
    env.vfs
        .put_file("/file", b"clean\nnew trailing \nend\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "git diff --check"),
        (
            2,
            "file:2: trailing whitespace.\n+new trailing \n".into(),
            String::new()
        )
    );
    assert_eq!(
        run(&mut env, "git add file; git diff --cached --check").0,
        2
    );
}

#[test]
fn tracked_files_can_be_moved_and_removed_atomically() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/old.txt", b"initial\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add old.txt; git commit -m initial").0, 0);

    env.vfs
        .put_file("/old.txt", b"modified\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git mv old.txt new.txt").0, 0);
    assert!(!env.vfs.exists("/", "/old.txt"));
    assert_eq!(env.vfs.read("/", "/new.txt").unwrap(), b"modified\n");
    assert_eq!(
        run(&mut env, "git status --short").1,
        "A  new.txt\nD  old.txt\n"
    );
    assert_eq!(run(&mut env, "git commit -m moved").0, 0);

    assert_eq!(run(&mut env, "git rm new.txt").0, 0);
    assert!(!env.vfs.exists("/", "/new.txt"));
    assert_eq!(run(&mut env, "git status --short").1, "D  new.txt\n");
    assert_eq!(run(&mut env, "git commit -m removed").0, 0);

    env.vfs
        .put_file("/cached.txt", b"cached\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add cached.txt").0, 0);
    assert_eq!(run(&mut env, "git rm --cached cached.txt").0, 0);
    assert!(env.vfs.exists("/", "/cached.txt"));
    assert_eq!(run(&mut env, "git status --short").1, "?? cached.txt\n");
}

#[test]
fn git_rm_refuses_modified_files_without_force() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/file", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m one").0, 0);
    env.vfs.put_file("/file", b"two\n".to_vec(), 0o644).unwrap();

    let refused = run(&mut env, "git rm file");
    assert_eq!(refused.0, 1);
    assert!(
        refused.2.contains("staged or local changes"),
        "{}",
        refused.2
    );
    assert!(env.vfs.exists("/", "/file"));
    assert_eq!(run(&mut env, "git rm -f file").0, 0);
    assert!(!env.vfs.exists("/", "/file"));

    env.vfs
        .put_file("/kept", b"kept\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add kept; git commit -m kept").0, 0);
    assert_ne!(run(&mut env, "git rm kept missing").0, 0);
    assert!(env.vfs.exists("/", "/kept"));
    assert_eq!(run(&mut env, "git status --short").1, "");
}

#[test]
fn compact_branch_status_and_diff_stat_support_agent_orientation() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/file", b"keep\nold\nend\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m base").0, 0);
    env.vfs
        .put_file("/file", b"keep\nnew\nextra\nend\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(run(&mut env, "git status -sb").1, "## main\n M file\n");
    assert_eq!(
        run(&mut env, "git diff --stat"),
        (
            0,
            " file | 3 ++-\n 1 file changed, 2 insertions(+), 1 deletion(-)\n".into(),
            String::new()
        )
    );
    assert_eq!(run(&mut env, "git add file; git diff --cached --stat").0, 0);
}

#[test]
fn global_options_select_the_repository_and_override_config() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "mkdir -p /work; git init -q /work").0, 0);
    env.vfs
        .put_file("/work/note.txt", b"hello\n".to_vec(), 0o644)
        .unwrap();

    // -C runs from another directory without changing the shell's own working directory.
    assert_eq!(run(&mut env, "git -C /work add --all").0, 0);
    assert_eq!(
        run(&mut env, "git -C /work status --porcelain").1,
        "A  note.txt\n"
    );
    assert_eq!(run(&mut env, "pwd").1, "/\n");

    let committed = run(
        &mut env,
        "git -C /work -c user.name=Ada -c user.email=ada@example.test commit -m recorded",
    );
    assert_eq!(committed.0, 0, "{}", committed.2);
    assert_eq!(
        run(&mut env, "git -C /work log -1 --format='%an <%ae>'").1,
        "Ada <ada@example.test>\n"
    );
    assert!(run(&mut env, "git --version").1.starts_with("git version "));
}

#[test]
fn add_all_accepts_paths_and_respects_gitignore() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/.gitignore", b"build/\n*.log\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/build/out.o", b"binary\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/run.log", b"noise\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/src/main.rs", b"fn main() {}\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(
        run(&mut env, "git status --porcelain").1,
        "?? .gitignore\n?? src/\n"
    );
    assert_eq!(run(&mut env, "git add -A .").0, 0);
    assert_eq!(run(&mut env, "git ls-files").1, ".gitignore\nsrc/main.rs\n");
    assert_eq!(
        run(&mut env, "git check-ignore run.log build/out.o").1,
        "run.log\nbuild/out.o\n"
    );
    // Ignored paths only reach the index when they are forced.
    assert_eq!(run(&mut env, "git add -f run.log").0, 0);
    assert!(run(&mut env, "git ls-files").1.contains("run.log"));
}

#[test]
fn long_status_matches_git_section_layout() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/tracked.txt", b"one\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "git add tracked.txt; git commit -m base").0,
        0
    );
    env.vfs
        .put_file("/tracked.txt", b"two\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/staged.txt", b"new\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/loose.txt", b"loose\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add staged.txt").0, 0);

    assert_eq!(
        run(&mut env, "git status").1,
        concat!(
            "On branch main\n",
            "Changes to be committed:\n",
            "  (use \"git restore --staged <file>...\" to unstage)\n",
            "\tnew file:   staged.txt\n",
            "\n",
            "Changes not staged for commit:\n",
            "  (use \"git add <file>...\" to update what will be committed)\n",
            "  (use \"git restore <file>...\" to discard changes in working directory)\n",
            "\tmodified:   tracked.txt\n",
            "\n",
            "Untracked files:\n",
            "  (use \"git add <file>...\" to include in what will be committed)\n",
            "\tloose.txt\n",
            "\n",
        )
    );
}

#[test]
fn renaming_a_tracked_file_is_reported_as_a_rename() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/old.txt", b"same\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add old.txt; git commit -m base").0, 0);
    assert_eq!(run(&mut env, "git mv old.txt new.txt").0, 0);
    assert_eq!(
        run(&mut env, "git status --porcelain").1,
        "R  old.txt -> new.txt\n"
    );
    assert!(run(&mut env, "git status")
        .1
        .contains("\trenamed:    old.txt -> new.txt\n"));
}

#[test]
fn diff_shows_hunks_with_context_and_ignores_untracked_files() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    let original = b"one\ntwo\nthree\nfour\nfive\nsix\nseven\n".to_vec();
    env.vfs.put_file("/f.txt", original, 0o644).unwrap();
    assert_eq!(run(&mut env, "git add f.txt; git commit -m base").0, 0);
    env.vfs
        .put_file(
            "/f.txt",
            b"one\ntwo\nthree\nFOUR\nfive\nsix\nseven\n".to_vec(),
            0o644,
        )
        .unwrap();
    env.vfs
        .put_file("/untracked.txt", b"ignore me\n".to_vec(), 0o644)
        .unwrap();

    let diff = run(&mut env, "git diff");
    assert_eq!(diff.0, 0, "{}", diff.2);
    assert!(diff.1.contains("@@ -1,7 +1,7 @@\n"), "{}", diff.1);
    assert!(diff.1.contains("-four\n+FOUR\n"), "{}", diff.1);
    assert!(!diff.1.contains("untracked.txt"), "{}", diff.1);
    assert!(!diff.1.contains("+seven"), "{}", diff.1);

    // An untracked file alone leaves the tree "unchanged" for --quiet, as in Git.
    assert_eq!(
        run(&mut env, "git checkout -- f.txt; git diff --quiet").0,
        0
    );
    assert_eq!(run(&mut env, "git diff nosuchrev").0, 128);
}

#[test]
fn diff_compares_revisions_and_ranges() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"first\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f.txt; git commit -m one").0, 0);
    env.vfs
        .put_file("/f.txt", b"second\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am two").0, 0);

    for command in ["git diff HEAD~1 HEAD", "git diff HEAD~1..HEAD"] {
        let diff = run(&mut env, command);
        assert_eq!(diff.0, 0, "{command}: {}", diff.2);
        assert!(
            diff.1.contains("-first\n+second\n"),
            "{command}: {}",
            diff.1
        );
    }
    assert_eq!(run(&mut env, "git diff --name-only HEAD~1").1, "f.txt\n");
}

#[test]
fn log_supports_ranges_formats_and_rejects_unknown_placeholders() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/f.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add f.txt; git commit -m one").0, 0);
    env.vfs.put_file("/f.txt", b"b\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git commit -am two").0, 0);

    assert_eq!(run(&mut env, "git log --oneline").1.lines().count(), 2);
    assert_eq!(
        run(&mut env, "git log --oneline HEAD~1..HEAD")
            .1
            .lines()
            .count(),
        1
    );
    assert_eq!(run(&mut env, "git log --format=%s").1, "two\none\n");
    assert_eq!(run(&mut env, "git log -1 --format=%ci").1.trim().len(), 25);

    // The default format carries the author and date lines agents parse.
    let default = run(&mut env, "git log -1").1;
    assert!(default.starts_with("commit "), "{default}");
    assert!(default.contains("\nAuthor: "), "{default}");
    assert!(default.contains("\nDate:   "), "{default}");

    let rejected = run(&mut env, "git log --pretty=nonsense");
    assert_eq!(rejected.0, 2);
    assert!(
        rejected.2.contains("unsupported log format"),
        "{}",
        rejected.2
    );
}

#[test]
fn commit_reports_its_summary_and_supports_amend_and_stdin_messages() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/a.txt", b"one\n".to_vec(), 0o644)
        .unwrap();
    let first = run(&mut env, "git add a.txt; git commit -m initial");
    assert_eq!(first.0, 0, "{}", first.2);
    assert!(first.1.contains("(root-commit)"), "{}", first.1);
    assert!(
        first.1.contains(" 1 file changed, 1 insertion(+)\n"),
        "{}",
        first.1
    );
    assert!(
        first.1.contains(" create mode 100644 a.txt\n"),
        "{}",
        first.1
    );

    env.vfs
        .put_file("/a.txt", b"two\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam updated").0, 0);
    assert_eq!(run(&mut env, "git log -1 --format=%s").1, "updated\n");

    assert_eq!(
        run(&mut env, "printf 'from stdin\\n' | git commit --amend -F -").0,
        0
    );
    assert_eq!(run(&mut env, "git log -1 --format=%s").1, "from stdin\n");
    // Amending rewrites the tip rather than adding a commit.
    assert_eq!(run(&mut env, "git log --oneline").1.lines().count(), 2);

    // Nothing staged reports the working-tree status and fails, as Git does.
    let empty = run(&mut env, "git commit -m nothing");
    assert_eq!(empty.0, 1);
    assert!(empty.1.contains("nothing to commit"), "{}", empty.1);
}

#[test]
fn config_uses_git_ini_files_and_a_global_scope() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    assert_eq!(run(&mut env, "git config user.name Ada").0, 0);
    let stored = String::from_utf8(env.vfs.read("/", "/.git/config").unwrap()).unwrap();
    assert!(stored.contains("[user]\n\tname = Ada\n"), "{stored}");

    // A hand-written INI block is readable, which is how agents add a remote.
    assert_eq!(
        run(
            &mut env,
            "printf '[remote \"origin\"]\\n\\turl = https://example.test/r.git\\n' >> .git/config",
        )
        .0,
        0
    );
    assert_eq!(
        run(&mut env, "git config remote.origin.url").1,
        "https://example.test/r.git\n"
    );
    assert_eq!(run(&mut env, "git remote -v").1.lines().count(), 2);

    assert_eq!(
        run(&mut env, "git config --global user.email ada@example.test").0,
        0
    );
    assert_eq!(
        run(&mut env, "git config --global --list").1,
        "user.email=ada@example.test\n"
    );
    // Repository settings override the per-user file.
    assert_eq!(run(&mut env, "git config user.name").1, "Ada\n");
}

#[test]
fn switching_branches_handles_the_current_branch_and_missing_targets() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"base\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f.txt; git commit -m base").0, 0);

    let already = run(&mut env, "git checkout main");
    assert_eq!(already.0, 0, "{}", already.2);
    assert_eq!(already.2, "Already on 'main'\n");

    assert_eq!(run(&mut env, "git switch -c feature").0, 0);
    assert_eq!(run(&mut env, "git switch -").0, 0);
    assert_eq!(run(&mut env, "git branch --show-current").1, "main\n");

    let missing = run(&mut env, "git checkout nosuchbranch");
    assert_eq!(missing.0, 1);
    assert!(
        missing.2.contains("did not match any file(s)"),
        "{}",
        missing.2
    );
    // Remote-tracking branches do not exist in the simulation.
    assert_eq!(run(&mut env, "git branch -r").1, "");
}

#[test]
fn checking_a_path_out_of_a_revision_also_stages_it() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"old\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f.txt; git commit -m one").0, 0);
    env.vfs
        .put_file("/f.txt", b"new\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am two").0, 0);

    assert_eq!(run(&mut env, "git checkout HEAD~1 -- f.txt").0, 0);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"old\n");
    assert_eq!(run(&mut env, "git status --porcelain").1, "M  f.txt\n");
}

#[test]
fn stash_saves_and_restores_uncommitted_work() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"base\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f.txt; git commit -m base").0, 0);
    env.vfs
        .put_file("/f.txt", b"work in progress\n".to_vec(), 0o644)
        .unwrap();

    let saved = run(&mut env, "git stash");
    assert_eq!(saved.0, 0, "{}", saved.2);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"base\n");
    assert_eq!(run(&mut env, "git status --porcelain").1, "");
    assert!(run(&mut env, "git stash list")
        .1
        .starts_with("stash@{0}: WIP on main:"));

    assert_eq!(run(&mut env, "git stash pop").0, 0);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"work in progress\n");
    assert_eq!(run(&mut env, "git stash list").1, "");
}

#[test]
fn merge_fast_forwards_and_refuses_content_conflicts() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/shared.txt", b"base\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add shared.txt; git commit -m base").0, 0);

    assert_eq!(run(&mut env, "git switch -c feature").0, 0);
    env.vfs
        .put_file("/added.txt", b"feature\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "git add added.txt; git commit -m feature").0,
        0
    );
    assert_eq!(run(&mut env, "git switch main").0, 0);

    let fast_forward = run(&mut env, "git merge feature");
    assert_eq!(fast_forward.0, 0, "{}", fast_forward.2);
    assert!(
        fast_forward.1.contains("Fast-forward\n"),
        "{}",
        fast_forward.1
    );
    assert_eq!(env.vfs.read("/", "/added.txt").unwrap(), b"feature\n");
    assert!(run(&mut env, "git merge feature")
        .1
        .contains("Already up to date."));

    // Diverging edits to one file are refused rather than written as conflict markers.
    assert_eq!(
        run(&mut env, "git switch -c other HEAD~1; git switch other").0,
        0
    );
    env.vfs
        .put_file("/shared.txt", b"theirs\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am theirs").0, 0);
    assert_eq!(run(&mut env, "git switch main").0, 0);
    env.vfs
        .put_file("/shared.txt", b"ours\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am ours").0, 0);

    let conflicted = run(&mut env, "git merge other");
    assert_eq!(conflicted.0, 1);
    assert!(
        conflicted.2.contains("Merge conflict in shared.txt"),
        "{}",
        conflicted.2
    );
    assert_eq!(env.vfs.read("/", "/shared.txt").unwrap(), b"ours\n");
}

#[test]
fn inspection_commands_read_objects_history_and_content() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/src/lib.rs", b"pub fn find() {}\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/README.md", b"docs\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m base").0, 0);

    assert_eq!(
        run(&mut env, "git grep -n find").1,
        "src/lib.rs:1:pub fn find() {}\n"
    );
    assert_eq!(run(&mut env, "git grep -l find").1, "src/lib.rs\n");
    assert_eq!(run(&mut env, "git grep missing").0, 1);
    assert_eq!(run(&mut env, "git show HEAD:README.md").1, "docs\n");
    assert_eq!(run(&mut env, "git cat-file -t HEAD").1, "commit\n");
    assert_eq!(
        run(&mut env, "git ls-tree --name-only HEAD").1,
        "README.md\nsrc\n"
    );
    assert_eq!(
        run(&mut env, "git ls-tree -r --name-only HEAD").1,
        "README.md\nsrc/lib.rs\n"
    );
    // Blob ids match a real repository's, so hash-object output is comparable.
    assert_eq!(
        run(&mut env, "printf 'a\\nb\\n' | git hash-object --stdin").1,
        "422c2b7ab3b3c668038da977e4e93a5fc623169c\n"
    );

    assert_eq!(run(&mut env, "git tag -a v1 -m 'first release'").0, 0);
    assert_eq!(run(&mut env, "git describe").1, "v1\n");
    assert!(run(&mut env, "git tag -n").1.contains("first release"));
    assert_eq!(run(&mut env, "git merge-base HEAD HEAD").1.trim().len(), 40);
}

#[test]
fn clean_removes_untracked_directories_only_with_d() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/tracked.txt", b"keep\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m base").0, 0);
    env.vfs
        .put_file("/loose.txt", b"drop\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/junk/inner.txt", b"drop\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(run(&mut env, "git clean -n").1, "Would remove loose.txt\n");
    assert_eq!(
        run(&mut env, "git clean -nd").1,
        "Would remove junk/\nWould remove loose.txt\n"
    );
    // Refusing without -f or -n protects an agent from an accidental wipe.
    assert_eq!(run(&mut env, "git clean").0, 128);
    assert_eq!(run(&mut env, "git clean -fd").0, 0);
    assert!(!env.vfs.exists("/", "/junk"));
    assert!(!env.vfs.exists("/", "/loose.txt"));
    assert_eq!(env.vfs.read("/", "/tracked.txt").unwrap(), b"keep\n");
}
