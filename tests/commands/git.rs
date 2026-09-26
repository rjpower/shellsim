//! Integration coverage for the deterministic, VFS-only `git` porcelain subset.
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
    assert_eq!(unknown.0, 129);
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
    assert_eq!(rejected.0, 128);
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
fn stash_reapply_refuses_to_write_over_an_uncommitted_edit() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a.txt", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -m one").0, 0);
    env.vfs
        .put_file("/a.txt", b"stashed\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git stash push -q -m saved").0, 0);
    assert_eq!(run(&mut env, "git stash push -q -m saved").1, "");

    // The entry and the working file are two versions that nothing has recorded, and the file can
    // only hold one. Git refuses rather than pick, and leaves the working file alone.
    env.vfs
        .put_file("/a.txt", b"conflict\n".to_vec(), 0o644)
        .unwrap();
    let refused = run(&mut env, "git stash pop");
    assert_eq!(refused.0, 1);
    assert!(
        refused.2.starts_with(
            "error: Your local changes to the following files would be overwritten by merge:\n\ta.txt\n"
        ),
        "{}",
        refused.2
    );
    assert_eq!(env.vfs.read("/", "/a.txt").unwrap(), b"conflict\n");
    // The entry survives a reapplication that did not happen.
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

    // The two changes touch with no unchanged line between them, which Git treats as a conflict
    // rather than picking an order for them.
    assert_eq!(run(&mut env, "git stash pop").0, 1);
    assert_eq!(
        env.vfs.read("/", "/f.txt").unwrap(),
        b"one\n<<<<<<< Updated upstream\ntwo\nCOMMITTED\n=======\nSTASHED\nthree\n>>>>>>> Stashed changes\n"
    );
    assert_eq!(run(&mut env, "git status --short").1, "UU f.txt\n");
    // The entry stays on the list so the conflict can be settled and the pop tried again.
    assert!(run(&mut env, "git stash list").1.contains("saved"));
}

#[test]
fn a_stash_that_touches_a_different_part_of_a_file_reapplies_cleanly() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f.txt", b"one\ntwo\nthree\nfour\nfive\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    env.vfs
        .put_file(
            "/f.txt",
            b"one\nSTASHED\nthree\nfour\nfive\n".to_vec(),
            0o644,
        )
        .unwrap();
    assert_eq!(run(&mut env, "git stash push -q -m saved").0, 0);

    env.vfs
        .put_file(
            "/f.txt",
            b"one\ntwo\nthree\nfour\nCOMMITTED\n".to_vec(),
            0o644,
        )
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam later").0, 0);

    assert_eq!(run(&mut env, "git stash pop").0, 0);
    assert_eq!(
        env.vfs.read("/", "/f.txt").unwrap(),
        b"one\nSTASHED\nthree\nfour\nCOMMITTED\n"
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
    let omitted = run(&mut env, "git bisect start");
    assert_eq!(omitted.0, 129);
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
fn commit_identity_from_the_environment_needs_an_exported_variable() {
    // `git` now runs as a genuine child process, so it sees exported variables only, the same
    // as any other external command. A shell variable that was never exported does not reach it;
    // an inline prefix assignment or an explicit `export` does, because both put the variable in
    // the command's environment.
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/one.txt", b"one\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(
        run(
            &mut env,
            "GIT_AUTHOR_NAME=Nobody; git add -A; git commit -m unexported"
        )
        .0,
        0
    );
    let unexported = run(&mut env, "git cat-file -p HEAD").1;
    assert!(
        unexported.contains("\nauthor shellsim <shellsim@localhost>"),
        "{unexported}"
    );

    env.vfs
        .put_file("/two.txt", b"two\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(
        run(
            &mut env,
            "git add -A; GIT_AUTHOR_NAME=Ada GIT_AUTHOR_EMAIL=ada@example.com git commit -m prefixed"
        )
        .0,
        0
    );
    let prefixed = run(&mut env, "git cat-file -p HEAD").1;
    assert!(
        prefixed.contains("\nauthor Ada <ada@example.com>"),
        "{prefixed}"
    );
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
    assert_eq!(run(&mut env, "git log --since='last tuesday'").0, 129);
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

#[test]
fn a_conflict_can_be_settled_by_taking_one_side() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f", b"l1\nl2\nl3\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc side").0, 0);
    env.vfs
        .put_file("/f", b"l1\nSIDE\nl3\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam side").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs
        .put_file("/f", b"l1\nMAIN\nl3\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam main").0, 0);
    assert_eq!(run(&mut env, "git merge side").0, 1);

    // The conflict is visible to the usual review commands.
    assert_eq!(run(&mut env, "git diff --name-status").1, "U\tf\n");
    assert_eq!(
        run(&mut env, "git diff --diff-filter=U --name-only").1,
        "f\n"
    );
    assert!(run(&mut env, "git diff").1.contains("+<<<<<<< HEAD"));

    assert_eq!(run(&mut env, "git checkout --theirs f").0, 0);
    assert_eq!(env.vfs.read("/", "/f").unwrap(), b"l1\nSIDE\nl3\n");
    // Taking a side does not by itself mark the path resolved.
    assert_eq!(run(&mut env, "git status --short").1, "UU f\n");

    assert_eq!(run(&mut env, "git restore --ours f").0, 0);
    assert_eq!(env.vfs.read("/", "/f").unwrap(), b"l1\nMAIN\nl3\n");
    assert_eq!(run(&mut env, "git add f; git commit -qm merged").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "");
}

#[test]
fn a_replay_leaves_other_staged_work_alone() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/a", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc side").0, 0);
    env.vfs.put_file("/s", b"s\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm side").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs
        .put_file("/keep", b"keep\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add keep").0, 0);

    // Committing the replay would fold the staged file into it, so Git refuses.
    let refused = run(&mut env, "git cherry-pick side");
    assert_eq!(refused.0, 128);
    assert!(refused.2.contains("would be overwritten"), "{}", refused.2);
    assert_eq!(run(&mut env, "git status --short").1, "A  keep\n");

    // `-n` commits nothing, so it goes ahead and leaves the staged file staged.
    assert_eq!(run(&mut env, "git cherry-pick -n side").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "A  keep\nA  s\n");
    // Nothing is recorded as in progress, so there is nothing to abort.
    assert_eq!(run(&mut env, "git cherry-pick --abort").0, 128);
    assert_eq!(run(&mut env, "git status --short").1, "A  keep\nA  s\n");
}

#[test]
fn a_merge_commit_names_both_parents_in_the_log() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/f", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc side").0, 0);
    env.vfs.put_file("/s", b"s\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm side").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs.put_file("/m", b"m\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm main").0, 0);
    let first = run(&mut env, "git rev-parse --short HEAD")
        .1
        .trim()
        .to_string();
    let second = run(&mut env, "git rev-parse --short side")
        .1
        .trim()
        .to_string();
    assert_eq!(run(&mut env, "git merge -m merged side").0, 0);

    assert!(
        run(&mut env, "git log -1")
            .1
            .contains(&format!("Merge: {first} {second}\n")),
        "{}",
        run(&mut env, "git log -1").1
    );
    // An ordinary commit has no such line.
    assert!(!run(&mut env, "git log -1 HEAD~1").1.contains("Merge:"));
}

#[test]
fn symbolic_links_are_tracked_as_their_targets() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    assert_eq!(run(&mut env, "echo target > t.txt; ln -s t.txt l.txt").0, 0);
    assert_eq!(
        run(&mut env, "git status --short").1,
        "?? l.txt\n?? t.txt\n"
    );

    assert_eq!(run(&mut env, "git add -A").0, 0);
    // Git stores a link as a blob holding its target, under mode 120000.
    assert!(
        run(&mut env, "git ls-files -s")
            .1
            .starts_with("120000 3eddab3ca20c14aaf1b71e59b3c4633f167afcc1 0\tl.txt\n"),
        "{}",
        run(&mut env, "git ls-files -s").1
    );
    let committed = run(&mut env, "git commit -m links");
    assert!(
        committed.1.contains(" create mode 120000 l.txt\n"),
        "{}",
        committed.1
    );

    // The link comes back as a link, not as a copy of what it points at.
    assert_eq!(run(&mut env, "rm l.txt").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, " D l.txt\n");
    assert_eq!(run(&mut env, "git checkout -- l.txt").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "");
    assert!(run(&mut env, "ls -l l.txt").1.contains("l.txt -> t.txt"));

    // Retargeting it is an ordinary content change.
    assert_eq!(run(&mut env, "rm l.txt; ln -s other.txt l.txt").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, " M l.txt\n");
    let patch = run(&mut env, "git diff").1;
    assert!(patch.contains("-t.txt"), "{patch}");
    assert!(patch.contains("+other.txt"), "{patch}");
}

#[test]
fn diff_and_log_accept_the_flags_agents_pass_by_habit() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f", b"a\nb  \nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc side").0, 0);
    env.vfs.put_file("/s", b"s\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm side").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs.put_file("/m", b"m\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm main").0, 0);
    assert_eq!(run(&mut env, "git merge -m merged side").0, 0);

    assert_eq!(
        run(&mut env, "git log --merges --oneline")
            .1
            .lines()
            .count(),
        1
    );
    assert_eq!(
        run(&mut env, "git log --no-merges --oneline")
            .1
            .lines()
            .count(),
        3
    );
    // Renames are always detected, so asking for them is accepted and changes nothing.
    assert_eq!(run(&mut env, "git diff -M HEAD~1 --name-only").0, 0);
    assert_eq!(run(&mut env, "git diff -M50% HEAD~1 --name-only").0, 0);

    // Under -w a context line is shown as it reads now, not as it read before.
    env.vfs
        .put_file("/f", b"a\nb\nCHANGED\n".to_vec(), 0o644)
        .unwrap();
    let patch = run(&mut env, "git diff -w").1;
    assert!(patch.contains("\n b\n"), "{patch:?}");
}

#[test]
fn the_reflog_records_where_head_has_been_and_brings_a_reset_back() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/f", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm one").0, 0);
    env.vfs.put_file("/f", b"two\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git commit -qam two").0, 0);
    assert_eq!(run(&mut env, "git switch -qc side").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    assert_eq!(run(&mut env, "git reset -q --hard HEAD~1").0, 0);

    let log = run(&mut env, "git reflog").1;
    let actions: Vec<&str> = log
        .lines()
        .map(|line| line.split_once(": ").map(|parts| parts.1).unwrap_or(line))
        .collect();
    assert_eq!(
        actions,
        vec![
            "reset: moving to HEAD~1",
            "checkout: moving from side to main",
            "checkout: moving from main to side",
            "commit: two",
            "commit (initial): one",
        ]
    );
    assert_eq!(run(&mut env, "git reflog -n 2").1.lines().count(), 2);

    // The commit the reset threw away is still reachable through the log.
    assert_eq!(env.vfs.read("/", "/f").unwrap(), b"one\n");
    assert_eq!(run(&mut env, "git reset -q --hard HEAD@{1}").0, 0);
    assert_eq!(env.vfs.read("/", "/f").unwrap(), b"two\n");
}

#[test]
fn rebase_replays_a_branch_onto_another() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs.put_file("/f", b"base\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc feature").0, 0);
    env.vfs.put_file("/a", b"a\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm f1").0, 0);
    env.vfs.put_file("/b", b"b\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm f2").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs.put_file("/m", b"m\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm m1").0, 0);
    assert_eq!(run(&mut env, "git switch -q feature").0, 0);

    assert_eq!(run(&mut env, "git rebase main").0, 0);
    let subjects: Vec<String> = run(&mut env, "git log --format=%s")
        .1
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(subjects, vec!["f2", "f1", "m1", "base"]);
    assert_eq!(run(&mut env, "git status --short").1, "");
    // Everything both branches wrote is present.
    for path in ["/a", "/b", "/m", "/f"] {
        assert!(env.vfs.exists("/", path), "{path} is missing");
    }
    assert!(run(&mut env, "git rebase main").1.contains("up to date"));
}

#[test]
fn a_conflicting_rebase_stops_and_can_be_continued_or_abandoned() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init -q").0, 0);
    env.vfs
        .put_file("/f", b"l1\nl2\nl3\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add -A; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc feature").0, 0);
    env.vfs
        .put_file("/f", b"l1\nFEATURE\nl3\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam f1").0, 0);
    env.vfs
        .put_file("/f", b"l1\nFEATURE\nl3\nextra\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam f2").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs
        .put_file("/f", b"l1\nMAIN\nl3\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam m1").0, 0);
    assert_eq!(run(&mut env, "git switch -q feature").0, 0);

    let stopped = run(&mut env, "git rebase main");
    assert_eq!(stopped.0, 1);
    assert!(stopped.2.contains("could not apply"), "{}", stopped.2);
    assert_eq!(run(&mut env, "git status --short").1, "UU f\n");

    // Abandoning it puts the branch back exactly as it was.
    assert_eq!(run(&mut env, "git rebase --abort").0, 0);
    assert_eq!(
        env.vfs.read("/", "/f").unwrap(),
        b"l1\nFEATURE\nl3\nextra\n"
    );
    assert_eq!(run(&mut env, "git log --format=%s").1, "f2\nf1\nbase\n");
    assert_eq!(run(&mut env, "git status --short").1, "");

    // Settling the conflict carries the rest of the branch over it.
    assert_eq!(run(&mut env, "git rebase main").0, 1);
    env.vfs
        .put_file("/f", b"l1\nBOTH\nl3\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f").0, 0);
    assert_eq!(run(&mut env, "git rebase --continue").0, 0);
    assert_eq!(run(&mut env, "git log --format=%s").1, "f2\nf1\nm1\nbase\n");
    assert_eq!(env.vfs.read("/", "/f").unwrap(), b"l1\nBOTH\nl3\nextra\n");
    assert_eq!(run(&mut env, "git status --short").1, "");
    assert_eq!(run(&mut env, "git rebase --continue").0, 128);
}

#[test]
fn two_branches_that_both_append_conflict_instead_of_looping() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f.txt; git add -A; git commit -q -m base; \
                 git switch -qc topic; printf 'a\\nTOPIC\\n' > f.txt; git commit -qam t1; \
                 git switch -q main; printf 'a\\nMAIN\\n' > f.txt; git commit -qam m1";
    assert_eq!(run(&mut env, setup).0, 0);

    let merge = run(&mut env, "git merge topic");
    assert_eq!(merge.0, 1, "{}", merge.2);
    assert!(
        merge
            .2
            .contains("CONFLICT (content): Merge conflict in f.txt"),
        "{}",
        merge.2
    );
    assert_eq!(
        run(&mut env, "cat f.txt").1,
        "a\n<<<<<<< HEAD\nMAIN\n=======\nTOPIC\n>>>>>>> topic\n"
    );
    assert_eq!(run(&mut env, "git status --short").1, "UU f.txt\n");
}

#[test]
fn blame_survives_a_file_that_grew_and_then_changed() {
    let mut env = Environment::new();
    let setup =
        "git init -q; printf 'one\\ntwo\\nthree\\n' > f.txt; git add -A; git commit -q -m c1; \
                 printf 'one\\ntwo\\nthree\\nfour\\n' > f.txt; git commit -qam c2; \
                 printf 'one\\ntwo\\nTHREE\\nfour\\n' > f.txt; git commit -qam c3";
    assert_eq!(run(&mut env, setup).0, 0);
    let first = run(&mut env, "git log --format=%H --reverse | head -1").1;
    let (c1, c2, c3) = {
        let ids = run(&mut env, "git log --format=%H --reverse").1;
        let mut lines = ids.lines();
        (
            lines.next().unwrap().to_string(),
            lines.next().unwrap().to_string(),
            lines.next().unwrap().to_string(),
        )
    };
    assert_eq!(first.trim(), c1);

    let blame = run(&mut env, "git blame f.txt");
    assert_eq!(blame.0, 0, "{}", blame.2);
    let attributed: Vec<&str> = blame
        .1
        .lines()
        .map(|line| line.split_whitespace().next().unwrap_or_default())
        .collect();
    // The two untouched lines came from the first commit, "four" from the append, "THREE" from
    // the edit that followed it.
    assert_eq!(
        attributed,
        [
            format!("^{}", &c1[..7]),
            format!("^{}", &c1[..7]),
            c3[..8].to_string(),
            c2[..8].to_string(),
        ]
    );
}

#[test]
fn a_rebase_refuses_to_start_on_work_that_is_not_committed() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'base\\n' > a; printf 'original\\n' > d; git add -A; \
                 git commit -q -m base; git switch -qc topic; printf 'topic\\n' > a; \
                 git commit -qam t1; git switch -q main; printf 'main\\n' > m; git add -A; \
                 git commit -qm m1; git switch -q topic; echo precious > d";
    assert_eq!(run(&mut env, setup).0, 0);

    let unstaged = run(&mut env, "git rebase main");
    assert_eq!(unstaged.0, 1, "{}", unstaged.2);
    assert!(
        unstaged
            .2
            .starts_with("error: cannot rebase: You have unstaged changes."),
        "{}",
        unstaged.2
    );
    assert_eq!(run(&mut env, "cat d").1, "precious\n");

    assert_eq!(run(&mut env, "git add d").0, 0);
    let staged = run(&mut env, "git rebase main");
    assert_eq!(staged.0, 1, "{}", staged.2);
    assert!(
        staged
            .2
            .starts_with("error: cannot rebase: Your index contains uncommitted changes."),
        "{}",
        staged.2
    );
    assert_eq!(run(&mut env, "cat d").1, "precious\n");
}

#[test]
fn a_stopped_rebase_leaves_every_branch_where_it_was() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'base\\n' > f; git add -A; git commit -q -m base; \
                 git switch -qc topic; printf 't1\\n' > f; git commit -qam t1; \
                 printf 't2\\n' > f; git commit -qam t2; git switch -q main; \
                 printf 'm1\\n' > f; git commit -qam m1; git switch -q topic";
    assert_eq!(run(&mut env, setup).0, 0);
    let topic_tip = run(&mut env, "git rev-parse topic").1;
    let main_tip = run(&mut env, "git rev-parse main").1;

    assert_eq!(run(&mut env, "git rebase main").0, 1);
    // Git detaches HEAD for the replay, so both branches still name what they named before.
    assert_eq!(run(&mut env, "git rev-parse topic").1, topic_tip);
    assert_eq!(run(&mut env, "git rev-parse main").1, main_tip);

    let switched = run(&mut env, "git switch main");
    assert_eq!(switched.0, 128, "{}", switched.2);
    assert!(
        switched.2.contains("cannot switch branch while rebasing"),
        "{}",
        switched.2
    );

    assert_eq!(run(&mut env, "git rebase --abort").0, 0);
    assert_eq!(run(&mut env, "git rev-parse topic").1, topic_tip);
    assert_eq!(run(&mut env, "git rev-parse main").1, main_tip);
    assert_eq!(
        run(&mut env, "git rev-parse --abbrev-ref HEAD").1,
        "topic\n"
    );
}

#[test]
fn moving_a_symbolic_link_stages_the_link_and_not_what_it_points_at() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'TARGET-CONTENT\\n' > t.txt; ln -s t.txt link; \
                 git add -A; git commit -q -m base";
    assert_eq!(run(&mut env, setup).0, 0);

    let moved = run(&mut env, "git mv link link2");
    assert_eq!(moved.0, 0, "{}", moved.2);
    // The blob is the link target text, which is what real Git stores for mode 120000.
    assert_eq!(
        run(&mut env, "git ls-files -s link2").1,
        "120000 3eddab3ca20c14aaf1b71e59b3c4633f167afcc1 0\tlink2\n"
    );
    assert_eq!(run(&mut env, "git status --short").1, "R  link -> link2\n");
    assert_eq!(run(&mut env, "git commit -qm mv").0, 0);
    assert_eq!(run(&mut env, "rm link2; git checkout -- link2").0, 0);
    assert_eq!(run(&mut env, "readlink link2").1, "t.txt\n");
}

#[test]
fn stashing_is_refused_mid_conflict_and_restores_new_files_staged() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'base\\n' > f.txt; printf 'keep\\n' > g.txt; git add -A; \
                 git commit -q -m base; git switch -qc topic; printf 'TOPIC\\n' > f.txt; \
                 printf 'TOPIC-IMPORTANT\\n' > g.txt; git commit -qam t1; git switch -q main; \
                 printf 'MAIN\\n' > f.txt; git commit -qam m1";
    assert_eq!(run(&mut env, setup).0, 0);
    assert_eq!(run(&mut env, "git merge topic").0, 1);

    let refused = run(&mut env, "git stash push -m mid");
    assert_eq!(refused.0, 1, "{}", refused.2);
    assert!(refused.2.contains("f.txt: needs merge"), "{}", refused.2);
    // The merged side of the other file is still there, so the merge commit will carry it.
    assert_eq!(run(&mut env, "cat g.txt").1, "TOPIC-IMPORTANT\n");

    assert_eq!(run(&mut env, "git merge --abort").0, 0);
    assert_eq!(
        run(&mut env, "echo brandnew > new.txt; git add new.txt").0,
        0
    );
    assert_eq!(run(&mut env, "git stash push -m s").0, 0);
    assert_eq!(run(&mut env, "git stash pop").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "A  new.txt\n");
}

#[test]
fn status_names_the_operation_that_is_unfinished() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f; git add -A; git commit -qm base; \
                 git switch -qc topic; printf 'T\\n' > f; git commit -qam t1; \
                 git switch -q main; printf 'M\\n' > f; git commit -qam m1";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git cherry-pick topic").0, 1);
    let picking = run(&mut env, "git status").1;
    assert!(
        picking.contains("You are currently cherry-picking commit"),
        "{picking}"
    );
    assert!(
        picking.contains("  (fix conflicts and run \"git cherry-pick --continue\")\n"),
        "{picking}"
    );
    assert!(
        picking
            .contains("  (use \"git cherry-pick --abort\" to cancel the cherry-pick operation)\n"),
        "{picking}"
    );

    assert_eq!(run(&mut env, "printf 'R\\n' > f; git add f").0, 0);
    let settled = run(&mut env, "git status").1;
    assert!(
        settled.contains("  (all conflicts fixed: run \"git cherry-pick --continue\")\n"),
        "{settled}"
    );

    assert_eq!(run(&mut env, "git cherry-pick --abort").0, 0);
    assert_eq!(run(&mut env, "git rebase topic").0, 1);
    let rebasing = run(&mut env, "git status").1;
    assert!(
        rebasing.contains("You are currently rebasing branch 'main' on '"),
        "{rebasing}"
    );
    assert!(
        rebasing.contains("  (use \"git rebase --abort\" to check out the original branch)\n"),
        "{rebasing}"
    );
}

#[test]
fn orig_head_names_where_a_reset_came_from() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f; git add -A; git commit -qm c1; \
                 printf 'b\\n' > f; git commit -qam c2; printf 'c\\n' > f; git commit -qam c3";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git reset --hard HEAD~2").0, 0);
    assert_eq!(run(&mut env, "git log --oneline | wc -l").1, "1\n");

    let back = run(&mut env, "git reset --hard ORIG_HEAD");
    assert_eq!(back.0, 0, "{}", back.2);
    assert_eq!(run(&mut env, "git log --oneline | wc -l").1, "3\n");
    assert_eq!(run(&mut env, "cat f").1, "c\n");
}

#[test]
fn a_rebase_can_name_the_branch_to_move_and_a_replay_can_be_skipped() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f; git add -A; git commit -qm base; \
                 git switch -qc topic; printf 't\\n' > g; git add -A; git commit -qm t1; \
                 git switch -q main; printf 'm\\n' > h; git add -A; git commit -qm m1";
    assert_eq!(run(&mut env, setup).0, 0);

    // The branch to move need not be the one checked out.
    let rebased = run(&mut env, "git rebase main topic");
    assert_eq!(rebased.0, 0, "{}", rebased.2);
    assert_eq!(
        run(&mut env, "git rev-parse --abbrev-ref HEAD").1,
        "topic\n"
    );
    assert_eq!(
        run(&mut env, "git log --format=%s | tr '\\n' ' '").1,
        "t1 m1 base "
    );

    assert_eq!(
        run(&mut env, "printf 'Y\\n' > f; git commit -qam other").0,
        0
    );
    assert_eq!(
        run(
            &mut env,
            "git switch -q main; printf 'X\\n' > f; git commit -qam clash"
        )
        .0,
        0
    );
    assert_eq!(
        run(&mut env, "git switch -q topic; git cherry-pick main").0,
        1
    );
    let skipped = run(&mut env, "git cherry-pick --skip");
    assert_eq!(skipped.0, 0, "{}", skipped.2);
    assert_eq!(run(&mut env, "git status --short").1, "");
    assert_eq!(run(&mut env, "cat f").1, "Y\n");
}

#[test]
fn blame_follows_a_file_that_was_renamed() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'one\\ntwo\\n' > a.txt; git add -A; git commit -qm c1; \
                 printf 'one\\nTWO\\n' > a.txt; git commit -qam c2; \
                 git mv a.txt b.txt; git commit -qm rename";
    assert_eq!(run(&mut env, setup).0, 0);
    let ids = run(&mut env, "git log --format=%H --reverse").1;
    let mut lines = ids.lines();
    let c1 = lines.next().unwrap().to_string();
    let c2 = lines.next().unwrap().to_string();

    let blame = run(&mut env, "git blame b.txt");
    assert_eq!(blame.0, 0, "{}", blame.2);
    let attributed: Vec<&str> = blame
        .1
        .lines()
        .map(|line| line.split_whitespace().next().unwrap_or_default())
        .collect();
    assert_eq!(attributed, [format!("^{}", &c1[..7]), c2[..8].to_string()]);

    // A path the working tree no longer has is still blamable when `--` names it.
    let old = run(&mut env, "git blame HEAD~1 -- a.txt");
    assert_eq!(old.0, 0, "{}", old.2);
    assert!(old.1.ends_with(" TWO\n"), "{}", old.1);
}

#[test]
fn short_status_lists_tracked_paths_in_path_order() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f.txt; printf 'k\\n' > g.txt; git add -A; \
                 git commit -qm base; git switch -qc topic; printf 'T\\n' > g.txt; \
                 git commit -qam t1; git switch -q main; printf 'M\\n' > g.txt; \
                 git commit -qam m1";
    assert_eq!(run(&mut env, setup).0, 0);
    assert_eq!(run(&mut env, "git merge topic").0, 1);
    assert_eq!(
        run(
            &mut env,
            "printf 'edited\\n' > f.txt; printf 'new\\n' > a.txt"
        )
        .0,
        0
    );

    // The unmerged path sorts among the others rather than being listed ahead of them.
    assert_eq!(
        run(&mut env, "git status --short").1,
        " M f.txt\nUU g.txt\n?? a.txt\n"
    );
}

#[test]
fn moving_between_commits_keeps_untracked_files_and_staged_work() {
    let mut env = Environment::new();
    let setup = "git init -q; echo base > b.txt; git add -A; git commit -qm c1; \
                 git switch -qc side; echo 'FROM SIDE' > n.txt; git add -A; git commit -qm s1; \
                 git switch -q main";
    assert_eq!(run(&mut env, setup).0, 0);

    // An untracked file is recorded nowhere, so nothing may write over it.
    for command in [
        "git checkout side",
        "git switch side",
        "git merge side",
        "git rebase side",
    ] {
        assert_eq!(run(&mut env, "echo PRECIOUS > n.txt").0, 0);
        let attempt = run(&mut env, command);
        assert_eq!(attempt.0, 1, "{command}: {}", attempt.2);
        assert!(
            attempt.2.starts_with(
                "error: The following untracked working tree files would be overwritten by"
            ),
            "{command}: {}",
            attempt.2
        );
        assert_eq!(run(&mut env, "cat n.txt").1, "PRECIOUS\n");
        assert_eq!(run(&mut env, "rm n.txt").0, 0);
    }

    // Staged work the move does not touch stays staged.
    assert_eq!(run(&mut env, "echo staged > s.txt; git add s.txt").0, 0);
    assert_eq!(run(&mut env, "git switch side").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "A  s.txt\n");
    assert_eq!(
        run(&mut env, "git diff --cached --name-status").1,
        "A\ts.txt\n"
    );
}

#[test]
fn a_hard_reset_ends_a_conflicted_merge() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\nb\\nc\\n' > f.txt; git add -A; git commit -qm c1; \
                 git switch -qc side; printf 'S\\nb\\nc\\n' > f.txt; git commit -qam s; \
                 git switch -q main; printf 'M\\nb\\nc\\n' > f.txt; git commit -qam m";
    assert_eq!(run(&mut env, setup).0, 0);
    assert_eq!(run(&mut env, "git merge side").0, 1);

    assert_eq!(run(&mut env, "git reset --hard").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "");
    assert_eq!(run(&mut env, "git ls-files -u").1, "");
    // The merge is over, so the next commit has one parent rather than being wedged.
    assert_eq!(
        run(
            &mut env,
            "echo later > later.txt; git add -A; git commit -qm after"
        )
        .0,
        0
    );
    assert_eq!(
        run(&mut env, "git log --format=%p -1")
            .1
            .split_whitespace()
            .count(),
        1
    );
}

#[test]
fn an_untracked_file_set_aside_comes_back_untracked() {
    let mut env = Environment::new();
    let setup = "git init -q; echo a > a.txt; git add -A; git commit -qm c1; \
                 echo junk > scratch.log; mkdir -p untr/nested; echo u > untr/nested/u.txt";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git stash push -u -m w").0, 0);
    assert_eq!(run(&mut env, "git stash pop").0, 0);
    assert_eq!(
        run(&mut env, "git status --short").1,
        "?? scratch.log\n?? untr/\n"
    );
    // Nothing was staged, so a sweeping commit does not pick the scratch files up.
    assert_eq!(run(&mut env, "git commit -qam next").0, 1);
}

#[test]
fn a_replay_of_several_commits_carries_on_past_a_conflict() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f.txt; git add -A; git commit -qm c1; \
                 git switch -qc feat; printf 'FEAT\\n' > f.txt; git add -A; git commit -qm p1; \
                 echo second > second.txt; git add -A; git commit -qm p2; \
                 git switch -q main; printf 'MAIN\\n' > f.txt; git add -A; git commit -qm m1";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git cherry-pick feat~1 feat").0, 1);
    assert_eq!(run(&mut env, "printf 'RES\\n' > f.txt; git add f.txt").0, 0);
    let finished = run(&mut env, "git cherry-pick --continue");
    assert_eq!(finished.0, 0, "{}", finished.2);
    // The second commit of the list is applied rather than silently dropped.
    assert_eq!(
        run(&mut env, "git log --format=%s | tr '\\n' ' '").1,
        "p2 p1 m1 c1 "
    );
    assert_eq!(run(&mut env, "cat second.txt").1, "second\n");
    assert_eq!(run(&mut env, "git status --short").1, "");
}

#[test]
fn abandoning_a_replay_undoes_the_commits_it_already_made() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f.txt; git add -A; git commit -qm c1; \
                 git switch -qc feat; echo first > one.txt; git add -A; git commit -qm p1; \
                 printf 'FEAT\\n' > f.txt; git add -A; git commit -qm p2; \
                 git switch -q main; printf 'MAIN\\n' > f.txt; git add -A; git commit -qm m1";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git cherry-pick feat~1 feat").0, 1);
    assert_eq!(
        run(&mut env, "git log --format=%s | tr '\\n' ' '").1,
        "p1 m1 c1 "
    );

    assert_eq!(run(&mut env, "git cherry-pick --abort").0, 0);
    assert_eq!(
        run(&mut env, "git log --format=%s | tr '\\n' ' '").1,
        "m1 c1 "
    );
    assert_eq!(run(&mut env, "git status --short").1, "");
    assert_eq!(run(&mut env, "test -e one.txt; echo $?").1, "1\n");
}

#[test]
fn an_empty_revision_names_what_is_staged() {
    let mut env = Environment::new();
    let setup = "git init -q; echo V1 > a.txt; git add -A; git commit -qm c1; \
                 echo V2 > a.txt; git add a.txt; echo V3 > a.txt";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git show :a.txt").1, "V2\n");
    assert_eq!(run(&mut env, "git show :0:a.txt").1, "V2\n");
    assert_eq!(run(&mut env, "git cat-file -p :a.txt").1, "V2\n");
    assert_eq!(run(&mut env, "git show HEAD:a.txt").1, "V1\n");
    assert_eq!(run(&mut env, "cat a.txt").1, "V3\n");
    assert_ne!(
        run(&mut env, "git rev-parse :a.txt").1,
        run(&mut env, "git rev-parse HEAD:a.txt").1
    );
}

#[test]
fn a_merge_follows_a_file_the_other_side_renamed() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'x\\ny\\nz\\n' > g.txt; echo pad > pad.txt; git add -A; \
                 git commit -qm c1; git switch -qc side; git mv g.txt gside.txt; \
                 git commit -qm rename; git switch -q main; printf 'x\\nMOD\\nz\\n' > g.txt; \
                 git add -A; git commit -qm modify";
    assert_eq!(run(&mut env, setup).0, 0);

    // The edit belongs under the new name; reporting a modify/delete conflict would lose it.
    let merged = run(&mut env, "git merge --no-edit side");
    assert_eq!(merged.0, 0, "{}", merged.2);
    assert_eq!(run(&mut env, "git status --short").1, "");
    assert_eq!(run(&mut env, "cat gside.txt").1, "x\nMOD\nz\n");
    assert_eq!(run(&mut env, "test -e g.txt; echo $?").1, "1\n");
    assert!(
        merged.1.contains(" rename g.txt => gside.txt (100%)\n"),
        "{}",
        merged.1
    );
}

#[test]
fn replaying_a_merge_needs_the_parent_it_is_measured_against() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f.txt; git add -A; git commit -qm base; \
                 git switch -qc side; echo s > s.txt; git add -A; git commit -qm s1; \
                 git switch -q main; echo m > m.txt; git add -A; git commit -qm m1; \
                 git merge --no-ff --no-edit side";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git switch -qc later main~1").0, 0);
    let bare = run(&mut env, "git cherry-pick main");
    assert_eq!(bare.0, 128, "{}", bare.2);
    assert!(
        bare.2.contains("is a merge but no -m option was given"),
        "{}",
        bare.2
    );

    let named = run(&mut env, "git cherry-pick -m 1 main");
    assert_eq!(named.0, 0, "{}", named.2);
    assert_eq!(run(&mut env, "cat s.txt").1, "s\n");

    // A plain commit has no parent to choose between.
    let wrong = run(&mut env, "git cherry-pick -m 1 side");
    assert_eq!(wrong.0, 128, "{}", wrong.2);
    assert!(wrong.2.contains("is not a merge"), "{}", wrong.2);
}

#[test]
fn a_branch_cannot_be_left_in_the_middle_of_an_operation() {
    let mut env = Environment::new();
    let setup = "git init -q; printf 'a\\n' > f.txt; git add -A; git commit -qm base; \
                 git switch -qc side; printf 'S\\n' > f.txt; git commit -qam s1; \
                 git switch -q main; printf 'M\\n' > f.txt; git commit -qam m1; \
                 git switch -qc other; git switch -q main";
    assert_eq!(run(&mut env, setup).0, 0);

    assert_eq!(run(&mut env, "git merge side").0, 1);
    let merging = run(&mut env, "git switch other");
    assert_eq!(merging.0, 128, "{}", merging.2);
    assert!(
        merging.2.contains("cannot switch branch while merging"),
        "{}",
        merging.2
    );
    assert_eq!(run(&mut env, "git merge --abort").0, 0);

    // Even with the conflict settled, the operation itself is still open.
    assert_eq!(run(&mut env, "git cherry-pick side").0, 1);
    assert_eq!(run(&mut env, "printf 'R\\n' > f.txt; git add f.txt").0, 0);
    let picking = run(&mut env, "git switch other");
    assert_eq!(picking.0, 128, "{}", picking.2);
    assert!(
        picking
            .2
            .contains("cannot switch branch while cherry-picking"),
        "{}",
        picking.2
    );
}

#[test]
fn a_mistyped_option_and_a_failed_operation_exit_differently() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/a", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add a; git commit -m one").0, 0);

    // Git exits 129 for an option it does not have, and 128 for work it could not do.
    let mistyped = run(&mut env, "git status --bogus");
    assert_eq!(mistyped.0, 129, "{}", mistyped.2);
    let missing = run(&mut env, "git add nosuch");
    assert_eq!(missing.0, 128, "{}", missing.2);
    let unresolvable = run(&mut env, "git restore --source=nope a");
    assert_eq!(unresolvable.0, 128, "{}", unresolvable.2);

    // Adding nothing is a no-op with a hint, not a failure.
    let nothing = run(&mut env, "git add");
    assert_eq!(nothing.0, 0, "{}", nothing.2);
    assert!(nothing.2.contains("Nothing specified"), "{}", nothing.2);
}

#[test]
fn the_first_commit_of_a_history_can_be_reverted() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/f", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add f; git commit -qm one").0, 0);

    // A root commit has no parent, so it is measured against the empty tree rather than refused.
    let reverted = run(&mut env, "git revert --no-edit HEAD");
    assert_eq!(reverted.0, 0, "{}", reverted.2);
    assert!(
        reverted.1.contains("delete mode 100644 f"),
        "{}",
        reverted.1
    );
    assert!(!env.vfs.exists("/", "/f"));
}

#[test]
fn a_rebase_names_what_it_did_in_the_reflog_and_reports_the_commit_it_settled() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/f", b"a\nb\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f; git commit -qm base").0, 0);
    assert_eq!(run(&mut env, "git switch -qc side").0, 0);
    env.vfs
        .put_file("/f", b"a\nB\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam side").0, 0);
    assert_eq!(run(&mut env, "git switch -q main").0, 0);
    env.vfs
        .put_file("/f", b"A\nb\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git commit -qam main").0, 0);

    let stopped = run(&mut env, "git rebase side");
    assert_eq!(stopped.0, 1, "{}", stopped.2);
    env.vfs
        .put_file("/f", b"A\nB\nc\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add f").0, 0);
    let finished = run(&mut env, "git rebase --continue");
    assert_eq!(finished.0, 0, "{}", finished.2);
    // Git names the commit the user settled by hand, and only that one.
    assert!(finished.1.contains("[detached HEAD"), "{}", finished.1);
    assert!(finished.1.contains("1 file changed"), "{}", finished.1);

    let reflog = run(&mut env, "git reflog -n 3").1;
    assert!(
        reflog.contains("rebase (finish): returning to refs/heads/main"),
        "{reflog}"
    );
    assert!(reflog.contains("rebase (continue): main"), "{reflog}");
    assert!(reflog.contains("rebase (start): checkout side"), "{reflog}");
}

#[test]
fn blame_can_print_the_whole_commit_id() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/f", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add f; git commit -qm one").0, 0);

    let short = run(&mut env, "git blame f").1;
    let long = run(&mut env, "git blame -l f").1;
    let id = run(&mut env, "git rev-parse HEAD").1.trim().to_string();
    assert!(short.starts_with(&format!("^{}", &id[..7])), "{short}");
    assert!(long.starts_with(&format!("^{}", &id[..39])), "{long}");
}

#[test]
fn a_one_line_log_entry_is_not_separated_from_its_diff() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/f", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add f; git commit -qm one").0, 0);

    // A one-line header has no message to separate from the diff, so Git runs them together.
    let oneline = run(&mut env, "git log --oneline --stat").1;
    assert!(oneline.contains("one\n f | 1 +\n"), "{oneline}");
    // The default header still has its blank line.
    let medium = run(&mut env, "git log --stat").1;
    assert!(medium.contains("    one\n\n f | 1 +\n"), "{medium}");
}

#[test]
fn a_corrupt_index_stops_the_commands_that_would_write_over_it() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/f", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add f; git commit -qm one").0, 0);
    env.vfs
        .put_file("/.git/index", b"not an index\n".to_vec(), 0o644)
        .unwrap();

    // An unreadable index used to read as "nothing is staged", which to `git clean` means every
    // tracked file is untracked and can be deleted.
    let cleaned = run(&mut env, "git clean -f");
    assert_eq!(cleaned.0, 128, "{}", cleaned.2);
    assert!(cleaned.2.contains("index file corrupt"), "{}", cleaned.2);
    assert!(env.vfs.exists("/", "/f"));

    for command in ["git add f", "git status", "git commit -m two", "git stash"] {
        let refused = run(&mut env, command);
        assert_eq!(refused.0, 128, "{command}: {}", refused.2);
    }
}

#[test]
fn a_missing_index_is_nothing_staged_rather_than_an_error() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/f", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add f; git commit -qm one").0, 0);
    env.vfs.remove_file("/", "/.git/index").unwrap();

    // Git reads an absent index file as an empty one, so the tracked file reads as staged for
    // deletion and untracked at once, which is what real Git prints here.
    let status = run(&mut env, "git status --short");
    assert_eq!(status.0, 0, "{}", status.2);
    assert_eq!(status.1, "D  f\n?? f\n");
}
