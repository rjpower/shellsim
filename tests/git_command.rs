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
        refused.2.contains("has local modifications"),
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
fn merge_fast_forwards_and_marks_content_conflicts() {
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

    // Diverging edits to one file are written as conflict markers for the user to settle.
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
    let marked = String::from_utf8(env.vfs.read("/", "/shared.txt").unwrap()).unwrap();
    assert_eq!(
        marked,
        "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> other\n"
    );
    assert_eq!(run(&mut env, "git status --short").1, "UU shared.txt\n");
    assert!(run(&mut env, "git status").1.contains("both modified:"));
    assert_eq!(
        run(&mut env, "git ls-files -u").1,
        "100644 df967b96a579e45a18b8251732d16804b2e56a55 1\tshared.txt\n\
         100644 b19a1e93bec1317dc6097229e12afaffbfa74dc2 2\tshared.txt\n\
         100644 950b81b7eee953d050aa05a641f8e056c85dd1bd 3\tshared.txt\n"
    );

    // Committing is refused until the path is staged, which is how resolution is recorded.
    assert_eq!(run(&mut env, "git commit -m merged").0, 1);
    env.vfs
        .put_file("/shared.txt", b"settled\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add shared.txt").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "M  shared.txt\n");
    assert_eq!(run(&mut env, "git commit -m merged").0, 0);
    assert_eq!(
        run(&mut env, "git log -1 --format=%P").1.split(' ').count(),
        2
    );
    assert_eq!(run(&mut env, "git status --short").1, "");
}

#[test]
fn a_merge_conflict_can_be_abandoned_without_losing_other_work() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/shared.txt", b"base\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m base").0, 0);
    assert_eq!(run(&mut env, "git switch -c other").0, 0);
    env.vfs
        .put_file("/shared.txt", b"theirs\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am theirs").0, 0);
    assert_eq!(run(&mut env, "git switch main").0, 0);
    env.vfs
        .put_file("/shared.txt", b"ours\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am ours").0, 0);
    env.vfs
        .put_file("/scratch.txt", b"not git's business\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(run(&mut env, "git merge other").0, 1);
    assert_eq!(run(&mut env, "git merge --abort").0, 0);
    assert_eq!(env.vfs.read("/", "/shared.txt").unwrap(), b"ours\n");
    assert_eq!(
        env.vfs.read("/", "/scratch.txt").unwrap(),
        b"not git's business\n"
    );
    assert_eq!(run(&mut env, "git status --short").1, "?? scratch.txt\n");
    assert_eq!(run(&mut env, "git merge --abort").0, 128);
}

#[test]
fn cherry_pick_and_revert_replay_one_commit() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"one\ntwo\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m base").0, 0);
    assert_eq!(run(&mut env, "git switch -c feature").0, 0);
    env.vfs
        .put_file("/f.txt", b"one\ntwo\nthree\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am three").0, 0);
    assert_eq!(run(&mut env, "git switch main").0, 0);

    assert_eq!(run(&mut env, "git cherry-pick feature").0, 0);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"one\ntwo\nthree\n");
    assert_eq!(run(&mut env, "git log --oneline").1.lines().count(), 2);

    assert_eq!(run(&mut env, "git revert --no-edit HEAD").0, 0);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"one\ntwo\n");
    assert!(run(&mut env, "git log -1 --format=%s")
        .1
        .starts_with("Revert \"three\""));
    assert_eq!(run(&mut env, "git status --short").1, "");
}

#[test]
fn a_conflicting_cherry_pick_stops_and_can_be_continued() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"one\ntwo\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m base").0, 0);
    assert_eq!(run(&mut env, "git switch -c feature").0, 0);
    env.vfs
        .put_file("/f.txt", b"feature\ntwo\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am feature").0, 0);
    assert_eq!(run(&mut env, "git switch main").0, 0);
    env.vfs
        .put_file("/f.txt", b"main\ntwo\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -am main").0, 0);

    let stopped = run(&mut env, "git cherry-pick feature");
    assert_eq!(stopped.0, 1);
    assert!(stopped.2.contains("could not apply"), "{}", stopped.2);
    assert_eq!(run(&mut env, "git status --short").1, "UU f.txt\n");
    assert_eq!(run(&mut env, "git cherry-pick --continue").0, 1);

    env.vfs
        .put_file("/f.txt", b"settled\ntwo\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f.txt").0, 0);
    assert_eq!(run(&mut env, "git cherry-pick --continue").0, 0);
    // A cherry-pick keeps one parent, unlike a merge.
    assert_eq!(
        run(&mut env, "git log -1 --format=%P").1.split(' ').count(),
        1
    );
    assert_eq!(run(&mut env, "git log -1 --format=%s").1, "feature\n");
    assert_eq!(run(&mut env, "git status --short").1, "");
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

#[test]
fn glob_pathspecs_select_files_at_any_depth() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/src/main.py", b"one\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/src/notes.md", b"one\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m base").0, 0);
    env.vfs
        .put_file("/src/main.py", b"two\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/src/notes.md", b"two\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(
        run(&mut env, "git status --porcelain -- '*.py'").1,
        " M src/main.py\n"
    );
    assert_eq!(
        run(&mut env, "git diff --name-only -- '*.md'").1,
        "src/notes.md\n"
    );
    assert_eq!(run(&mut env, "git ls-files '*.py'").1, "src/main.py\n");
}

#[test]
fn adding_an_ignored_file_by_name_is_refused() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/.gitignore", b"*.log\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/debug.log", b"noise\n".to_vec(), 0o644)
        .unwrap();

    let refused = run(&mut env, "git add debug.log");
    assert_eq!(refused.0, 1);
    assert!(
        refused.2.contains("ignored by one of your"),
        "{}",
        refused.2
    );
    assert_eq!(run(&mut env, "git status --short").1, "?? .gitignore\n");

    // `-v` names the rule that decided the path, and `-f` overrides it.
    assert_eq!(
        run(&mut env, "git check-ignore -v debug.log").1,
        ".gitignore:1:*.log\tdebug.log\n"
    );
    assert_eq!(run(&mut env, "git add -f debug.log").0, 0);
    assert!(run(&mut env, "git status --short")
        .1
        .contains("A  debug.log"));
}

#[test]
fn repository_excludes_and_character_classes_are_honoured() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/.git/info/exclude", b"scratch/\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/.gitignore", b"page[0-9].txt\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/scratch/tmp.bin", b"x\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/page1.txt", b"x\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/pages.txt", b"x\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(
        run(&mut env, "git status --short").1,
        "?? .gitignore\n?? pages.txt\n"
    );
    assert_eq!(run(&mut env, "git check-ignore scratch/tmp.bin").0, 0);
}

#[test]
fn checkout_keeps_edits_to_files_the_move_does_not_touch() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    env.vfs.put_file("/b.txt", b"b\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    assert_eq!(run(&mut env, "git branch side").0, 0);
    env.vfs
        .put_file("/b.txt", b"changed\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -a -m two").0, 0);

    // a.txt is identical on both branches, so the local edit survives the switch.
    env.vfs
        .put_file("/a.txt", b"local\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git switch side").0, 0);
    assert_eq!(env.vfs.read("/", "/a.txt").unwrap(), b"local\n");
    assert_eq!(env.vfs.read("/", "/b.txt").unwrap(), b"b\n");

    // A tracked file deleted from the working tree is restored even when its content is unchanged.
    assert_eq!(run(&mut env, "rm a.txt; git checkout HEAD -- .").0, 0);
    assert_eq!(env.vfs.read("/", "/a.txt").unwrap(), b"a\n");
}

#[test]
fn stash_reapply_merges_and_marks_what_it_cannot_settle() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/a.txt", b"stashed\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git stash push -q -m saved").0, 0);
    assert_eq!(run(&mut env, "git stash push -q -m saved").1, "");

    env.vfs
        .put_file("/a.txt", b"conflict\n".to_vec(), 0o644)
        .unwrap();
    let marked = run(&mut env, "git stash pop");
    assert_eq!(marked.0, 1);
    assert!(marked.2.contains("Merge conflict in a.txt"), "{}", marked.2);
    assert_eq!(
        String::from_utf8(env.vfs.read("/", "/a.txt").unwrap()).unwrap(),
        "<<<<<<< Updated upstream\nconflict\n=======\nstashed\n>>>>>>> Stashed changes\n"
    );
    assert_eq!(run(&mut env, "git status --short").1, "UU a.txt\n");
    // The entry survives a reapplication that did not finish.
    assert!(run(&mut env, "git stash list").1.contains("saved"));
}

#[test]
fn stash_reapply_keeps_work_committed_in_the_meantime() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"one\ntwo\nthree\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    env.vfs
        .put_file("/f.txt", b"one\nSTASHED\nthree\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git stash push -q -m saved").0, 0);

    // A change to a different part of the same file is committed while the work is set aside.
    env.vfs
        .put_file("/f.txt", b"one\ntwo\nCOMMITTED\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam later").0, 0);

    assert_eq!(run(&mut env, "git stash pop").0, 0);
    assert_eq!(
        env.vfs.read("/", "/f.txt").unwrap(),
        b"one\nSTASHED\nCOMMITTED\n"
    );
    // Git restores the working tree only, leaving the user to stage again.
    assert_eq!(run(&mut env, "git status --short").1, " M f.txt\n");
    assert_eq!(run(&mut env, "git stash list").1, "");
}

#[test]
fn committing_a_pathspec_records_only_those_files() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    env.vfs.put_file("/b.txt", b"b\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/a.txt", b"edited\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/b.txt", b"edited\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(run(&mut env, "git commit -m two b.txt").0, 0);
    assert_eq!(run(&mut env, "git status --porcelain").1, " M a.txt\n");
    assert_eq!(
        run(&mut env, "git show --name-only --format=%s HEAD").1,
        "two\n\nb.txt\n"
    );

    // A dry run reports the long status and fails when nothing is staged.
    let dry = run(&mut env, "git commit --dry-run");
    assert_eq!(dry.0, 1);
    assert!(dry.1.contains("Changes not staged for commit"), "{}", dry.1);
}

#[test]
fn removing_the_last_file_removes_its_directory() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/pkg/mod/unit.txt", b"x\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/keep.txt", b"k\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);

    assert_eq!(run(&mut env, "git rm -r -q pkg").0, 0);
    assert!(!env.vfs.exists("/", "/pkg"));
    assert!(env.vfs.exists("/", "/keep.txt"));
}

#[test]
fn inspection_commands_name_trees_and_revisions() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/src/main.py", b"print(1)\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/README.md", b"docs\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);

    let body = run(&mut env, "git cat-file -p HEAD").1;
    assert!(body.starts_with("tree "), "{body}");
    assert!(
        body.contains("\ncommitter shellsim <shellsim@localhost>"),
        "{body}"
    );
    // `-s` reports the size of exactly those bytes.
    assert_eq!(
        run(&mut env, "git cat-file -s HEAD")
            .1
            .trim()
            .parse::<usize>(),
        Ok(body.len())
    );

    // A tree id is shared by `ls-tree`, `cat-file`, and `rev-parse`.
    let listed = run(&mut env, "git ls-tree -d HEAD").1;
    assert!(listed.starts_with("040000 tree "), "{listed}");
    assert!(listed.ends_with("\tsrc\n"), "{listed}");
    assert_eq!(
        run(&mut env, "git rev-parse 'HEAD^{tree}'").1,
        body.lines().next().unwrap()["tree ".len()..].to_string() + "\n"
    );

    assert_eq!(
        run(&mut env, "git rev-parse HEAD:README.md").1,
        run(&mut env, "git hash-object README.md").1
    );
    assert_eq!(
        run(&mut env, "git rev-parse --symbolic-full-name HEAD").1,
        "refs/heads/main\n"
    );
    assert_eq!(
        run(&mut env, "git rev-parse 'HEAD^{commit}'").1,
        run(&mut env, "git rev-parse HEAD").1
    );
}

#[test]
fn grep_reports_paths_relative_to_the_working_directory() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/src/a.txt", b"hello world\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/docs/b.txt", b"hello docs\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);

    assert_eq!(
        run(&mut env, "git grep -n hello").1,
        "docs/b.txt:1:hello docs\nsrc/a.txt:1:hello world\n"
    );
    assert_eq!(
        run(&mut env, "(cd src && git grep -n hello)").1,
        "a.txt:1:hello world\n"
    );
    // A revision operand searches that commit and prefixes each match with it.
    env.vfs
        .put_file("/src/a.txt", b"goodbye\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "git grep -n hello HEAD").1,
        "HEAD:docs/b.txt:1:hello docs\nHEAD:src/a.txt:1:hello world\n"
    );
}

#[test]
fn unknown_subcommands_are_reported_the_way_git_reports_them() {
    let mut env = Environment::new();
    let unknown = run(&mut env, "git frobnicate");
    assert_eq!(unknown.0, 1);
    assert_eq!(
        unknown.2,
        "git: 'frobnicate' is not a git command. See 'git --help'.\n"
    );
    // A real Git subcommand this subset leaves out says so instead.
    let omitted = run(&mut env, "git rebase main");
    assert_eq!(omitted.0, 2);
    assert!(
        omitted.2.contains("unsupported subcommand"),
        "{}",
        omitted.2
    );
}

#[test]
fn log_filters_by_message_author_and_content() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/a.txt", b"alpha\nbeta\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m 'add alpha'").0, 0);
    env.vfs
        .put_file("/a.txt", b"alpha\ngamma\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m 'swap beta'").0, 0);
    env.vfs
        .put_file("/a.txt", b"delta\ngamma\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m 'drop alpha'").0, 0);

    assert_eq!(
        run(&mut env, "git log --grep=alpha --format=%s").1,
        "drop alpha\nadd alpha\n"
    );
    assert_eq!(
        run(&mut env, "git log -i --grep=ALPHA --format=%s").1,
        "drop alpha\nadd alpha\n"
    );
    // `-S` selects commits that changed how often the string occurs.
    assert_eq!(
        run(&mut env, "git log -S alpha --format=%s").1,
        "drop alpha\nadd alpha\n"
    );
    // `-G` selects commits with a matching changed line.
    assert_eq!(
        run(&mut env, "git log -G 'beta|gamma' --format=%s").1,
        "swap beta\nadd alpha\n"
    );
    assert_eq!(run(&mut env, "git log --author nobody --format=%s").1, "");
}

#[test]
fn configuration_keeps_every_value_and_expands_aliases() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);

    assert_eq!(run(&mut env, "git config --add fetch.refspec one").0, 0);
    assert_eq!(run(&mut env, "git config --add fetch.refspec two").0, 0);
    assert_eq!(
        run(&mut env, "git config --get-all fetch.refspec").1,
        "one\ntwo\n"
    );
    // A plain read returns the last value, and a plain write replaces every value.
    assert_eq!(run(&mut env, "git config fetch.refspec").1, "two\n");
    assert_eq!(run(&mut env, "git config fetch.refspec only").0, 0);
    assert_eq!(
        run(&mut env, "git config --get-all fetch.refspec").1,
        "only\n"
    );

    assert_eq!(run(&mut env, "git config core.bare").1, "false\n");
    assert_eq!(
        run(&mut env, "git config --type=bool core.bare").1,
        "false\n"
    );
    assert!(run(&mut env, "git config --show-origin core.bare")
        .1
        .starts_with("file:"));

    env.vfs
        .put_file("/note.txt", b"one\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git config alias.st 'status --short'").0, 0);
    assert_eq!(run(&mut env, "git st").1, "?? note.txt\n");
}

#[test]
fn apply_can_record_a_patch_in_the_index() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"a\nb\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/f.txt", b"a\nB\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "git diff > /p.diff; git checkout -- f.txt").0,
        0
    );

    // `--cached` patches the index and leaves the working tree alone.
    assert_eq!(run(&mut env, "git apply --cached /p.diff").0, 0);
    assert_eq!(
        run(&mut env, "git status --short").1,
        "MM f.txt\n?? p.diff\n"
    );
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"a\nb\nc\n");

    assert_eq!(run(&mut env, "git reset -q").0, 0);
    // `--index` patches both.
    assert_eq!(run(&mut env, "git apply --index /p.diff").0, 0);
    assert_eq!(
        run(&mut env, "git status --short").1,
        "M  f.txt\n?? p.diff\n"
    );
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"a\nB\nc\n");

    // A patch that no longer applies says so in Git's words and changes nothing.
    let refused = run(&mut env, "git apply /p.diff");
    assert_eq!(refused.0, 1);
    assert!(refused.2.contains("patch does not apply"), "{}", refused.2);
}

#[test]
fn status_names_paths_relative_to_the_working_directory() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    env.vfs
        .put_file("/sub/b.txt", b"b\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/a.txt", b"edited\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/sub/b.txt", b"edited\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/new.txt", b"n\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(
        run(&mut env, "(cd sub && git status --short)").1,
        " M ../a.txt\n M b.txt\n?? ../new.txt\n"
    );
}

#[test]
fn branch_listing_shows_a_detached_head() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs.put_file("/a.txt", b"b\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git commit -a -m two").0, 0);
    let head = run(&mut env, "git rev-parse HEAD~1").1.trim().to_string();

    assert_eq!(run(&mut env, "git checkout -q HEAD~1").0, 0);
    assert_eq!(
        run(&mut env, "git branch").1,
        format!("* (HEAD detached at {})\n  main\n", &head[..7])
    );
}

#[test]
fn quiet_suppresses_the_report_but_not_the_diagnostic() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -q -m one").0, 0);
    assert_eq!(run(&mut env, "git branch side").0, 0);

    let switched = run(&mut env, "git checkout -q side");
    assert_eq!(
        (switched.0, switched.1.as_str(), switched.2.as_str()),
        (0, "", "")
    );
    let created = run(&mut env, "git switch -q -c other");
    assert_eq!(
        (created.0, created.1.as_str(), created.2.as_str()),
        (0, "", "")
    );

    // A failure still explains itself.
    let missing = run(&mut env, "git checkout -q nonexistent");
    assert_eq!(missing.0, 1);
    assert!(missing.2.contains("did not match"), "{}", missing.2);
}

#[test]
fn diff_reports_a_move_as_a_rename() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/old.txt", b"aaa\nbbb\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    assert_eq!(run(&mut env, "git mv old.txt new.txt").0, 0);

    assert_eq!(
        run(&mut env, "git diff --cached").1,
        "diff --git a/old.txt b/new.txt\nsimilarity index 100%\nrename from old.txt\nrename to new.txt\n"
    );
    assert_eq!(
        run(&mut env, "git diff --cached --name-status").1,
        "R100\told.txt\tnew.txt\n"
    );
    assert_eq!(
        run(&mut env, "git diff --cached --stat").1,
        " old.txt => new.txt | 0\n 1 file changed, 0 insertions(+), 0 deletions(-)\n"
    );
    assert_eq!(
        run(&mut env, "git diff --cached --numstat").1,
        "0\t0\told.txt => new.txt\n"
    );
    assert_eq!(
        run(&mut env, "git diff --cached --summary").1,
        " rename old.txt => new.txt (100%)\n"
    );
}

#[test]
fn diff_can_ignore_whitespace_and_work_outside_a_repository() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/a.rs", b"fn main() {\nlet x = 1;\n}\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/b.txt", b"keep\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/a.rs", b"fn main() {\n    let x = 1;\n}\n".to_vec(), 0o644)
        .unwrap();

    // Reindentation alone is not a change under `-w`.
    assert_eq!(run(&mut env, "git diff -w").1, "");
    assert_eq!(run(&mut env, "git diff -w --stat").1, "");
    assert_eq!(run(&mut env, "git diff -w --quiet").0, 0);
    // A real edit still shows, and the whitespace-only file stays out of it.
    env.vfs
        .put_file("/b.txt", b"changed\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git diff -w --name-only").1, "b.txt\n");

    // `--no-index` needs no repository and exits 1 when the files differ.
    let differing = run(&mut env, "git diff --no-index a.rs b.txt");
    assert_eq!(differing.0, 1);
    assert!(
        differing.1.starts_with("diff --git a/a.rs b/b.txt\n"),
        "{}",
        differing.1
    );
    assert_eq!(run(&mut env, "git diff --no-index a.rs a.rs").0, 0);
}

#[test]
fn reference_plumbing_reads_the_ref_store() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    assert_eq!(run(&mut env, "git tag v1; git branch side").0, 0);
    let head = run(&mut env, "git rev-parse HEAD").1.trim().to_string();

    assert_eq!(
        run(&mut env, "git show-ref").1,
        format!("{head} refs/heads/main\n{head} refs/heads/side\n{head} refs/tags/v1\n")
    );
    assert_eq!(
        run(&mut env, "git show-ref --heads").1,
        format!("{head} refs/heads/main\n{head} refs/heads/side\n")
    );
    assert_eq!(
        run(&mut env, "git symbolic-ref HEAD").1,
        "refs/heads/main\n"
    );
    assert_eq!(run(&mut env, "git symbolic-ref --short HEAD").1, "main\n");
    assert_eq!(
        run(
            &mut env,
            "git for-each-ref --format='%(refname:short)' refs/heads/"
        )
        .1,
        "main\nside\n"
    );
    assert_eq!(
        run(&mut env, "git branch --format='%(refname:short)'").1,
        "main\nside\n"
    );
    assert_eq!(run(&mut env, "git tag --points-at HEAD").1, "v1\n");
    assert_eq!(
        run(&mut env, "git tag -d v1").1,
        format!("Deleted tag 'v1' (was {})\n", &head[..7])
    );

    // A detached HEAD is not a symbolic reference.
    assert_eq!(run(&mut env, "git checkout -q HEAD").0, 0);
    assert_eq!(run(&mut env, "git symbolic-ref HEAD").0, 1);
}

#[test]
fn stash_labels_entries_and_can_show_them() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"a\nb\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/f.txt", b"a\nB\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(run(&mut env, "git stash push -q -m wip").0, 0);
    assert_eq!(
        run(&mut env, "git stash list").1,
        "stash@{0}: On main: wip\n"
    );
    assert_eq!(
        run(&mut env, "git stash show").1,
        " f.txt | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)\n"
    );
    assert!(run(&mut env, "git stash show -p").1.contains("-b\n+B\n"));
    assert!(run(&mut env, "git stash pop").0 == 0);
}

#[test]
fn caret_excludes_a_revision_from_the_history() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    assert_eq!(run(&mut env, "git branch base").0, 0);
    env.vfs.put_file("/a.txt", b"b\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git commit -a -m two").0, 0);
    env.vfs.put_file("/a.txt", b"c\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git commit -a -m three").0, 0);

    assert_eq!(
        run(&mut env, "git log --format=%s HEAD ^base").1,
        "three\ntwo\n"
    );
    assert_eq!(run(&mut env, "git rev-list --count HEAD ^base").1, "2\n");
}

#[test]
fn status_reports_the_second_porcelain_format() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    env.vfs.put_file("/k.txt", b"k\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/a.txt", b"a\nx\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "git add a.txt; git rm -q --cached k.txt").0,
        0
    );

    let head = run(&mut env, "git rev-parse HEAD:a.txt")
        .1
        .trim()
        .to_string();
    let staged = run(&mut env, "git hash-object a.txt").1.trim().to_string();
    let removed = run(&mut env, "git rev-parse HEAD:k.txt")
        .1
        .trim()
        .to_string();
    let missing = "0".repeat(40);
    assert_eq!(
        run(&mut env, "git status --porcelain=v2").1,
        format!(
            "1 M. N... 100644 100644 100644 {head} {staged} a.txt\n\
             1 D. N... 100644 000000 000000 {removed} {missing} k.txt\n\
             ? k.txt\n"
        )
    );
    assert!(run(&mut env, "git status --porcelain=v2 -b")
        .1
        .contains("# branch.head main\n"));
}

#[test]
fn apply_reports_and_reverses_a_patch() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"a\nb\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/f.txt", b"a\nB\nc\nd\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "git diff > /p.diff; git checkout -- f.txt").0,
        0
    );

    assert_eq!(
        run(&mut env, "git apply --stat /p.diff").1,
        " f.txt |    3 ++-\n 1 file changed, 2 insertions(+), 1 deletion(-)\n"
    );
    assert_eq!(
        run(&mut env, "git apply --numstat /p.diff").1,
        "2\t1\tf.txt\n"
    );

    assert_eq!(run(&mut env, "git apply /p.diff").0, 0);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"a\nB\nc\nd\n");
    // `-R` undoes exactly what the patch did.
    assert_eq!(run(&mut env, "git apply -R /p.diff").0, 0);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"a\nb\nc\n");
    // The patch can also arrive on standard input.
    assert_eq!(run(&mut env, "git apply < /p.diff").0, 0);
    assert_eq!(env.vfs.read("/", "/f.txt").unwrap(), b"a\nB\nc\nd\n");
}

#[test]
fn log_filters_by_date() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(
        run(
            &mut env,
            "git add -A; GIT_AUTHOR_DATE=2024-06-01T00:00:00Z git commit -m old"
        )
        .0,
        0
    );
    env.vfs.put_file("/a.txt", b"b\n".to_vec(), 0o644).unwrap();
    assert_eq!(
        run(
            &mut env,
            "git add -A; GIT_AUTHOR_DATE=2025-06-01T00:00:00Z git commit -m new"
        )
        .0,
        0
    );

    assert_eq!(
        run(&mut env, "git log --since=2025-01-01 --format=%s").1,
        "new\n"
    );
    assert_eq!(
        run(&mut env, "git log --until=2025-01-01 --format=%s").1,
        "old\n"
    );
    assert_eq!(
        run(
            &mut env,
            "git log --since 2024-01-01 --until 2024-12-31 --format=%s"
        )
        .1,
        "old\n"
    );
    // A date this subset cannot read is refused rather than guessed at.
    assert_eq!(run(&mut env, "git log --since='last tuesday'").0, 2);
    assert_eq!(
        run(&mut env, "git log -1 --format=%ad").1,
        "Sun Jun 1 00:00:00 2025 +0000\n"
    );
}

#[test]
fn stat_output_stays_within_the_column_budget() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    let long = "a/very/deep/nested/path/file_with_a_rather_long_name_1.txt";
    let body: Vec<u8> = (1..=20)
        .map(|n| format!("line {n}\n"))
        .collect::<String>()
        .into();
    env.vfs
        .put_file(&format!("/{long}"), body.clone(), 0o644)
        .unwrap();
    env.vfs.put_file("/s.txt", b"s\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    let more: Vec<u8> = (1..=40)
        .map(|n| format!("line {n}\n"))
        .collect::<String>()
        .into();
    env.vfs.put_file(&format!("/{long}"), more, 0o644).unwrap();
    env.vfs
        .put_file("/s.txt", b"s\nt\n".to_vec(), 0o644)
        .unwrap();

    // The name is elided at a directory boundary rather than widening the line.
    assert_eq!(
        run(&mut env, "git diff --stat").1,
        " .../nested/path/file_with_a_rather_long_name_1.txt   | 20 ++++++++++++++++++++\n \
         s.txt                                                |  1 +\n \
         2 files changed, 21 insertions(+)\n"
    );
}

#[test]
fn the_executable_bit_is_tracked_through_a_commit_and_a_checkout() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/s.sh", b"#!/bin/sh\necho ok\n".to_vec(), 0o755)
        .unwrap();
    assert_eq!(run(&mut env, "git add s.sh").0, 0);
    assert!(run(&mut env, "git ls-files -s").1.starts_with("100755 "));
    let committed = run(&mut env, "git commit -m script");
    assert!(
        committed.1.contains(" create mode 100755 s.sh\n"),
        "{}",
        committed.1
    );

    // Dropping the bit is a change in its own right, with no content to index.
    assert_eq!(run(&mut env, "chmod -x s.sh").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, " M s.sh\n");
    assert_eq!(
        run(&mut env, "git diff").1,
        "diff --git a/s.sh b/s.sh\nold mode 100755\nnew mode 100644\n"
    );

    // A checkout puts the recorded mode back, so the script runs again.
    assert_eq!(run(&mut env, "git checkout -- s.sh").0, 0);
    assert_eq!(run(&mut env, "test -x s.sh").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "");

    // The bit survives moving between branches.
    assert_eq!(run(&mut env, "git switch -qc other").0, 0);
    env.vfs
        .put_file("/s.sh", b"#!/bin/sh\necho other\n".to_vec(), 0o755)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam other").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    assert_eq!(run(&mut env, "test -x s.sh").0, 0);
    assert_eq!(run(&mut env, "git switch -q other").0, 0);
    assert_eq!(run(&mut env, "test -x s.sh").0, 0);
}

#[test]
fn log_graph_draws_a_branch_and_the_merge_that_closes_it() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/f", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc side").0, 0);
    env.vfs.put_file("/side", b"s\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm side").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs.put_file("/main", b"m\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm main").0, 0);
    assert_eq!(run(&mut env, "git merge -m merged side").0, 0);

    let shape: Vec<String> = run(&mut env, "git log --graph --oneline")
        .1
        .lines()
        .map(|line| {
            line.split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert_eq!(shape, vec!["*", "|\\", "*", "|", "|/", "*"]);
    assert!(run(&mut env, "git log --graph --oneline")
        .1
        .contains("| * "));
}

#[test]
fn blame_attributes_each_line_to_the_commit_that_wrote_it() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f", b"one\ntwo\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    env.vfs
        .put_file("/f", b"one\nTWO\nthree\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam next").0, 0);
    let base = run(&mut env, "git rev-parse HEAD~1").1;
    let next = run(&mut env, "git rev-parse HEAD").1;
    // The first commit is a boundary, which Git marks with `^` in place of a hash digit.
    assert_eq!(
        run(&mut env, "git blame -s f").1,
        format!(
            "^{} 1) one\n{next} 2) TWO\n{next} 3) three\n",
            &base[..7],
            next = &next[..8]
        )
    );
    assert!(run(&mut env, "git blame f").1.contains("(shellsim "));
    assert_eq!(run(&mut env, "git blame -L3,3 f").1.lines().count(), 1);

    // An edit that is not committed yet is attributed to nobody.
    env.vfs
        .put_file("/f", b"one\nTWO\nthree\nfour\n".to_vec(), 0o644)
        .unwrap();
    let pending = run(&mut env, "git blame f").1;
    assert!(pending.contains("00000000 (Not Committed Yet"), "{pending}");
    assert!(pending.trim_end().ends_with("four"), "{pending}");
}

#[test]
fn apply_refuses_a_patch_whose_file_has_moved_on() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f", b"a\nb\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    env.vfs
        .put_file("/f", b"a\nB\nc\nd\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git diff > p.patch").0, 0);
    assert_eq!(run(&mut env, "git checkout -- f").0, 0);

    // The patch's last hunk ran to the end of the file, so it no longer fits once the file grows.
    env.vfs
        .put_file("/f", b"a\nb\nc\nZZZ\n".to_vec(), 0o644)
        .unwrap();
    let checked = run(&mut env, "git apply --check p.patch");
    assert_eq!(checked.0, 1);
    assert!(checked.2.contains("does not apply"), "{}", checked.2);
    assert_eq!(run(&mut env, "git apply p.patch").0, 1);
    assert_eq!(env.vfs.read("/", "/f").unwrap(), b"a\nb\nc\nZZZ\n");

    // A hunk in the middle of a file still applies at an offset, as it does in Git.
    assert_eq!(run(&mut env, "git checkout -- f").0, 0);
    assert_eq!(run(&mut env, "git apply p.patch").0, 0);
    assert_eq!(env.vfs.read("/", "/f").unwrap(), b"a\nB\nc\nd\n");
}
