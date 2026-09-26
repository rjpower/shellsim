//! Compatibility tests for logical process identity, lifecycle, and synthetic pseudo-filesystems.
//!
//! The suite checks observable shell behavior and directly verifies that generated `/proc` state
//! neither enters nor mutates the persistent VFS.

use shellsim::{
    scheduler::TaskState,
    vfs::{NativeProgram, NodeKind},
    Environment, Limits, StopReason,
};

fn run(env: &mut Environment, source: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = env.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn proc_self_describes_the_active_logical_shell() {
    let mut env = Environment::new();
    let status = run(&mut env, "cat /proc/1234/status");
    assert_eq!(status.0, 0, "{}", status.2);
    assert!(status.1.contains("Name:\tbash\n"), "{}", status.1);
    assert!(status.1.contains("Pid:\t1234\n"), "{}", status.1);
    assert!(status.1.contains("PPid:\t0\n"), "{}", status.1);
    assert!(status.1.contains("NSpgid:\t1234\n"), "{}", status.1);
    assert!(status.1.contains("NSsid:\t1234\n"), "{}", status.1);

    let child_status = run(&mut env, "cat /proc/self/status");
    assert_eq!(child_status.0, 0, "{}", child_status.2);
    assert!(
        child_status.1.contains("PPid:\t1234\n"),
        "{}",
        child_status.1
    );
    assert!(
        !child_status.1.lines().any(|line| line == "Pid:\t1234"),
        "{}",
        child_status.1
    );

    let readlink_pid = run(&mut env, "readlink /proc/self").1;
    let readlink_pid = readlink_pid.trim().parse::<u32>().unwrap();
    assert_ne!(readlink_pid, 1_234);
    assert!(env
        .invocations
        .events()
        .iter()
        .any(|event| { event.pid == readlink_pid && event.argv == ["readlink", "/proc/self"] }));
    assert_eq!(run(&mut env, "readlink /proc/self/cwd").1, "/\n");
    assert_eq!(
        run(&mut env, "ls /proc/self/fd")
            .1
            .split_whitespace()
            .collect::<Vec<_>>(),
        vec!["0", "1", "2"]
    );
    assert!(run(&mut env, "readlink /proc/self/fd/1")
        .1
        .starts_with("pipe:["));
    let listing = run(&mut env, "ls /proc");
    assert_eq!(listing.0, 0, "{}", listing.2);
    assert!(listing.1.contains("1234"), "{}", listing.1);
    assert!(listing.1.contains("meminfo"), "{}", listing.1);
}

#[test]
fn ls_inspects_virtual_directory_metadata_and_symlinks() {
    let mut env = Environment::new();
    env.vfs.mkdir("/", "/work/listing").unwrap();
    env.vfs.mkdir("/", "/work/listing/sub").unwrap();
    env.vfs
        .write("/", "/work/listing/sub/data", b"sixish", 0o640)
        .unwrap();
    env.vfs
        .symlink("/", "sub/data", "/work/listing/link")
        .unwrap();
    let listing = run(&mut env, "cd /work; ls -l listing");
    assert_eq!(listing.0, 0, "{}", listing.2);
    assert!(listing.1.contains("link -> sub/data"), "{}", listing.1);
    assert!(listing.1.contains("sub"), "{}", listing.1);
    let recursive = run(&mut env, "ls -R /work/listing");
    assert_eq!(recursive.0, 0, "{}", recursive.2);
    assert!(
        recursive.1.contains("/work/listing/sub:\ndata\n"),
        "{}",
        recursive.1
    );
}

#[test]
fn native_argv_children_use_process_cwd_and_leave_the_shell_unchanged() {
    let mut env = Environment::new();
    assert_eq!(
        run(&mut env, "env -C /work pwd"),
        (0, "/work\n".into(), "".into())
    );
    assert!(env
        .invocations
        .events()
        .iter()
        .any(|event| { event.pid != 1_234 && event.argv == ["pwd"] && event.status == Some(0) }));
    assert_eq!(run(&mut env, "pwd"), (0, "/\n".into(), "".into()));
    assert_eq!(run(&mut env, "env false").0, 1);
    assert_eq!(run(&mut env, "env true").0, 0);
    let readlink_pid = run(&mut env, "readlink /proc/self").1;
    let readlink_pid = readlink_pid.trim().parse::<u32>().unwrap();
    assert_ne!(readlink_pid, 1_234);
    assert!(env
        .invocations
        .events()
        .iter()
        .any(|event| { event.pid == readlink_pid && event.argv == ["readlink", "/proc/self"] }));
}

#[test]
fn git_runs_as_a_native_child_process() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    assert_eq!(run(&mut env, "git status").0, 0);
    assert!(env.invocations.events().iter().any(|event| {
        event.pid != 1_234
            && event.argv.first().is_some_and(|arg| arg == "git")
            && event.status == Some(0)
    }));
}

#[test]
fn native_mkdir_uses_the_child_system_handle() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "env -C /work mkdir -p nested/leaf").0, 0);
    assert!(matches!(
        env.vfs
            .metadata("/", "/work/nested/leaf", true)
            .unwrap()
            .kind,
        NodeKind::Dir
    ));
    assert!(env.invocations.events().iter().any(|event| {
        event.pid != 1_234
            && event.argv == ["mkdir", "-p", "nested/leaf"]
            && event.status == Some(0)
    }));
    let failure = run(&mut env, "mkdir /missing/leaf");
    assert_eq!(failure.0, 1);
    assert!(
        failure.2.contains("cannot create directory"),
        "{}",
        failure.2
    );
    let unsupported = run(&mut env, "mkdir --mode=777 /work/nope");
    assert_eq!(unsupported.0, 2);
    assert!(
        unsupported.2.contains("unimplemented option"),
        "{}",
        unsupported.2
    );
    assert!(run(&mut env, "cat --help").2.contains("--help"));
}

#[test]
fn typed_registered_commands_run_in_child_processes() {
    let mut env = Environment::new();
    env.vfs.mkdir("/", "/work/typed").unwrap();
    env.vfs
        .write("/", "/work/typed/item", b"content", 0o644)
        .unwrap();

    assert_eq!(
        run(&mut env, "env -C /work ls typed"),
        (0, "item\n".into(), "".into())
    );
    assert_eq!(run(&mut env, "env -C /work mv typed/item typed/moved").0, 0);
    assert_eq!(run(&mut env, "chmod 600 /work/typed/moved").0, 0);
    assert_eq!(
        env.vfs
            .metadata("/", "/work/typed/moved", true)
            .unwrap()
            .mode
            & 0o777,
        0o600
    );
    assert_eq!(run(&mut env, "rmdir /work/typed").0, 1);
    assert_eq!(
        run(&mut env, "stat -c '%a:%s' /work/typed/moved"),
        (0, "600:7\n".into(), "".into())
    );
    assert_eq!(run(&mut env, "du -b /work/typed").1, "7\t/work/typed\n");
    assert!(run(&mut env, "tree /work/typed").1.contains("└── moved"));
    assert_eq!(run(&mut env, "basename /work/typed/moved").1, "moved\n");
    assert_eq!(
        run(&mut env, "dirname /work/typed/moved").1,
        "/work/typed\n"
    );
    assert_eq!(run(&mut env, "chmod -R 700 /work/absent").0, 1);
    env.vfs.remove_file("/", "/work/typed/moved").unwrap();
    assert_eq!(run(&mut env, "rmdir /work/typed").0, 0);
    for command in [
        "ls", "mv", "chmod", "rmdir", "stat", "du", "tree", "basename", "dirname",
    ] {
        assert!(
            env.invocations.events().iter().any(|event| {
                event.pid != 1_234
                    && event.argv.first().is_some_and(|arg| arg == command)
                    && event.status.is_some()
            }),
            "{command} did not run in a child process"
        );
    }
    env.vfs
        .copy_file("/", "/usr/bin/ls", "/work/list-typed")
        .unwrap();
    assert_eq!(run(&mut env, "/work/list-typed /work").0, 0);
    env.vfs.remove_file("/", "/usr/bin/ls").unwrap();
    assert_eq!(run(&mut env, "ls /work").0, 127);
}

#[test]
fn typed_registered_output_obeys_the_shared_limit() {
    let mut env = Environment::with_limits(Limits {
        cpu: 1_000_000,
        memory: 16 * 1024 * 1024,
        disk: 16 * 1024 * 1024,
        output: 8,
    });
    for name in ["first", "second", "third"] {
        env.vfs
            .write("/", &format!("/work/{name}"), b"", 0o644)
            .unwrap();
    }
    let (outcome, stdout, _) = env.run_script_capture("ls /work");
    assert_eq!(outcome.exit_status, 137);
    assert!(stdout.len() <= 8);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
}

#[test]
fn typed_registered_output_resumes_across_pipe_backpressure() {
    let mut env = Environment::new();
    env.vfs.mkdir("/", "/work/many").unwrap();
    let suffix = "x".repeat(90);
    for index in 0..800 {
        env.vfs
            .write("/", &format!("/work/many/{index:04}-{suffix}"), b"", 0o644)
            .unwrap();
    }
    assert_eq!(
        run(&mut env, "ls /work/many | wc -l"),
        (0, "800\n".into(), "".into())
    );
}

#[test]
fn typed_rm_removes_trees_without_crossing_the_virtual_root() {
    let mut env = Environment::new();
    env.vfs.mkdir_all("/", "/work/remove/sub").unwrap();
    env.vfs
        .write("/", "/work/remove/sub/file", b"data", 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "rm -r /work/remove").0, 0);
    assert!(env.vfs.metadata("/", "/work/remove", false).is_err());
    assert_eq!(run(&mut env, "rm -f /work/missing").0, 0);
    let root = run(&mut env, "rm -rf /");
    assert_eq!(root.0, 1, "{}", root.2);
    assert!(root.2.contains("refusing to remove virtual root"));
    assert!(env.vfs.metadata("/", "/work", false).is_ok());
}

#[test]
fn typed_file_metadata_commands_use_child_cwd_and_identity() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "env -C /work touch target").0, 0);
    assert_eq!(run(&mut env, "env -C /work ln -s target link").0, 0);
    assert_eq!(run(&mut env, "env -C /work readlink link").1, "target\n");
    assert_eq!(
        run(&mut env, "env -C /work realpath link").1,
        "/work/target\n"
    );
    assert_eq!(
        run(&mut env, "env -C /work realpath -m absent").1,
        "/work/absent\n"
    );
    assert_eq!(run(&mut env, "env -C /work chown 42 target").0, 0);
    assert_eq!(run(&mut env, "env -C /work chgrp 7 target").0, 0);
    assert_eq!(run(&mut env, "chgrp --unknown 7 /work/target").0, 2);
    assert_eq!(run(&mut env, "chgrp root:7 /work/target").0, 1);
    let target = env.vfs.metadata("/", "/work/target", false).unwrap();
    assert_eq!((target.uid, target.gid), (42, 7));
    assert_eq!(run(&mut env, "readlink /work/target").0, 1);
    for command in ["touch", "ln", "readlink", "realpath", "chown", "chgrp"] {
        assert!(
            env.invocations.events().iter().any(|event| {
                event.pid != 1_234
                    && event.argv.first().is_some_and(|arg| arg == command)
                    && event.status.is_some()
            }),
            "{command} did not run in a child process"
        );
    }
}

#[test]
fn typed_mktemp_creates_private_process_owned_entries() {
    let mut env = Environment::new();
    let file = run(&mut env, "mktemp note.XXXXXX");
    assert_eq!(file, (0, "/tmp/note.000000\n".into(), "".into()));
    let metadata = env.vfs.metadata("/", "/tmp/note.000000", false).unwrap();
    assert_eq!(metadata.mode & 0o777, 0o600);
    let directory = run(&mut env, "mktemp -d group.XXXXXX");
    assert_eq!(directory, (0, "/tmp/group.000001\n".into(), "".into()));
    assert!(matches!(
        env.vfs
            .metadata("/", "/tmp/group.000001", false)
            .unwrap()
            .kind,
        NodeKind::Dir
    ));
    assert!(env.invocations.events().iter().any(|event| {
        event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == "mktemp")
    }));
}

#[test]
fn typed_file_reads_virtual_files_and_rejects_unsupported_sources() {
    let mut env = Environment::new();
    env.vfs
        .write("/", "/work/words", b"hello\n", 0o644)
        .unwrap();
    env.vfs
        .write("/", "/work/binary", b"\xff\0", 0o644)
        .unwrap();
    assert_eq!(
        run(&mut env, "file /work/words /work/binary /work"),
        (
            0,
            "/work/words: ASCII text\n/work/binary: data\n/work: directory\n".into(),
            "".into(),
        )
    );
    let device = run(&mut env, "file /dev/zero");
    assert_eq!(device.0, 1);
    assert!(device.1.contains("cannot open"));
    assert!(env.invocations.events().iter().any(|event| {
        event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == "file")
    }));
}

#[test]
fn remaining_filesystem_tools_run_as_registered_native_children() {
    let mut env = Environment::new();
    env.vfs
        .write("/", "/work/source", b"payload", 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "cp /work/source /work/copied").0, 0);
    assert_eq!(run(&mut env, "truncate -s 3 /work/copied").0, 0);
    assert_eq!(env.vfs.read("/", "/work/copied").unwrap(), b"pay");
    assert_eq!(run(&mut env, "install -D /work/source /work/bin/tool").0, 0);
    assert_eq!(env.vfs.read("/", "/work/bin/tool").unwrap(), b"payload");
    assert_eq!(
        run(&mut env, "file /work/bin/tool").1,
        "/work/bin/tool: ASCII text\n"
    );
    assert_eq!(run(&mut env, "cp /usr/bin/pwd /work/copied-pwd").0, 0);
    assert_eq!(
        run(&mut env, "/work/copied-pwd"),
        (0, "/\n".into(), "".into())
    );
    for command in ["cp", "truncate", "install", "file"] {
        assert!(
            env.invocations.events().iter().any(|event| {
                event.pid != 1_234
                    && event.argv.first().is_some_and(|arg| arg == command)
                    && event.status == Some(0)
            }),
            "{command} did not run in a child process"
        );
    }
}

#[test]
fn copied_registered_image_keeps_its_original_path_entrypoint() {
    let mut env = Environment::new();
    env.vfs.write("/", "/work/owned", b"data", 0o644).unwrap();
    assert_eq!(run(&mut env, "cp /usr/bin/chgrp /work/reassigned").0, 0);
    env.vfs.remove_file("/", "/usr/bin/chgrp").unwrap();
    assert_eq!(run(&mut env, "/work/reassigned 42 /work/owned").0, 0);
    let metadata = env.vfs.metadata("/", "/work/owned", false).unwrap();
    assert_eq!((metadata.uid, metadata.gid), (0, 42));
    assert_eq!(run(&mut env, "chgrp 7 /work/owned").0, 127);
}

#[test]
fn stateless_commands_read_child_process_state_through_system() {
    let mut env = Environment::new();
    assert_eq!(
        run(&mut env, "/usr/bin/echo -n hello"),
        (0, "hello".into(), "".into())
    );
    assert_eq!(
        run(&mut env, "env -i ANSWER=42 printenv ANSWER"),
        (0, "42\n".into(), "".into())
    );
    assert_eq!(run(&mut env, "whoami"), (0, "root\n".into(), "".into()));
    assert_eq!(run(&mut env, "id -u"), (0, "0\n".into(), "".into()));
    assert_eq!(run(&mut env, "groups"), (0, "root\n".into(), "".into()));
    assert_eq!(run(&mut env, "uname -s"), (0, "Linux\n".into(), "".into()));
    assert_eq!(run(&mut env, "arch"), (0, "x86_64\n".into(), "".into()));
    assert_eq!(run(&mut env, "nproc"), (0, "1\n".into(), "".into()));
    assert_eq!(
        run(&mut env, "getconf PAGESIZE"),
        (0, "4096\n".into(), "".into())
    );
    assert_eq!(run(&mut env, "cd /work; df .").0, 0);
    assert_eq!(run(&mut env, "free -b").0, 0);
    assert_eq!(run(&mut env, "arch extra").0, 2);
    for command in [
        "/usr/bin/echo",
        "printenv",
        "whoami",
        "id",
        "groups",
        "uname",
        "arch",
        "nproc",
        "getconf",
        "df",
        "free",
    ] {
        assert!(
            env.invocations.events().iter().any(|event| {
                event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == command)
            }),
            "{command} did not run in a child process"
        );
    }
}

#[test]
fn native_images_are_opaque_vfs_executables_found_through_path() {
    let mut env = Environment::new();
    let node = env.vfs.metadata("/", "/usr/bin/pwd", true).unwrap();
    assert!(matches!(
        node.kind,
        NodeKind::NativeExecutable(NativeProgram::Pwd)
    ));
    assert_eq!(node.mode & 0o111, 0o111);
    assert!(env.vfs.read("/", "/usr/bin/pwd").is_err());
    env.vfs.copy_file("/", "/usr/bin/pwd", "/work/cwd").unwrap();
    assert_eq!(run(&mut env, "/work/cwd"), (0, "/\n".into(), "".into()));
    assert_eq!(run(&mut env, "which pwd").1, "/usr/bin/pwd\n");
    assert_eq!(run(&mut env, "/usr/bin/pwd"), (0, "/\n".into(), "".into()));
    assert_eq!(
        run(&mut env, "printf x | /usr/bin/pwd | wc -c"),
        (0, "2\n".into(), "".into())
    );
}

#[test]
fn native_yes_uses_path_and_process_scoped_pipe_backpressure() {
    let mut env = Environment::new();
    let node = env.vfs.metadata("/", "/usr/bin/yes", true).unwrap();
    assert!(matches!(
        node.kind,
        NodeKind::NativeExecutable(NativeProgram::Yes)
    ));
    assert_eq!(
        run(&mut env, "yes ready | head -n 3"),
        (0, "ready\nready\nready\n".into(), "".into())
    );
    assert!(
        env.invocations.events().iter().any(|event| {
            event.pid != 1_234 && event.argv == ["yes", "ready"] && event.status == Some(141)
        }),
        "{:?}",
        env.invocations.events()
    );

    env.vfs
        .copy_file("/", "/usr/bin/yes", "/work/repeat")
        .unwrap();
    assert_eq!(
        run(&mut env, "/work/repeat again | head -n 2").1,
        "again\nagain\n"
    );
    assert_eq!(run(&mut env, "env -i PATH=/missing yes").0, 127);
}

#[test]
fn registered_native_commands_resolve_only_from_executable_vfs_entries() {
    let mut env = Environment::new();
    assert!(matches!(
        env.vfs.metadata("/", "/usr/bin", true).unwrap().kind,
        NodeKind::Dir
    ));
    let node = env.vfs.metadata("/", "/usr/bin/cat", true).unwrap();
    assert!(matches!(
        node.kind,
        NodeKind::NativeExecutable(NativeProgram::Cat)
    ));
    assert_eq!(run(&mut env, "which cat").1, "/usr/bin/cat\n");
    assert_eq!(run(&mut env, "env -i PATH=/missing cat /work/note").0, 127);

    env.vfs.write("/", "/work/note", b"visible", 0o644).unwrap();
    env.vfs
        .copy_file("/", "/usr/bin/cat", "/work/reader")
        .unwrap();
    assert_eq!(run(&mut env, "/work/reader /work/note").1, "visible");

    env.vfs.remove_file("/", "/usr/bin/cat").unwrap();
    assert_eq!(run(&mut env, "cat /work/note").0, 127);
    assert_eq!(run(&mut env, "which cat").0, 1);
    assert_eq!(run(&mut env, "/work/reader /work/note").1, "visible");
}

#[test]
fn native_cat_streams_files_pipes_and_generated_devices() {
    let mut env = Environment::new();
    env.vfs.write("/", "/work/one", b"alpha\n", 0o644).unwrap();
    env.vfs.write("/", "/work/two", b"beta\n", 0o644).unwrap();
    assert_eq!(
        run(&mut env, "cat -n /work/one /work/two"),
        (0, "     1\talpha\n     2\tbeta\n".into(), String::new())
    );
    let missing = run(&mut env, "cat /work/one /work/missing /work/two");
    assert_eq!(missing.0, 1);
    assert_eq!(missing.1, "alpha\nbeta\n");
    assert!(missing.2.contains("/work/missing"), "{}", missing.2);
    assert_eq!(run(&mut env, "printf 'live\n' | cat").1, "live\n");
    assert_eq!(run(&mut env, "cat /dev/zero | head -c 8192").1.len(), 8192);
    assert_eq!(
        run(&mut env, "cat /dev/null"),
        (0, String::new(), String::new())
    );
    let unsupported = run(&mut env, "cat -z /work/one");
    assert_eq!(unsupported.0, 2);
    assert!(
        unsupported.2.contains("unimplemented option"),
        "{}",
        unsupported.2
    );
    assert!(env
        .invocations
        .events()
        .iter()
        .any(|event| { event.pid != 1_234 && event.argv.first().is_some_and(|arg| arg == "cat") }));
}

#[test]
fn native_cat_stops_at_the_virtual_output_limit() {
    let mut env = Environment::with_limits(Limits {
        cpu: 1_000_000,
        memory: 16 * 1024 * 1024,
        disk: 16 * 1024 * 1024,
        output: 128,
    });
    env.vfs
        .write("/", "/work/big", &[b'x'; 4096], 0o644)
        .unwrap();
    let (outcome, stdout, _) = env.run_script_capture("cat /work/big");
    assert_eq!(outcome.exit_status, 137);
    assert!(stdout.len() <= 128);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));

    let mut diagnostic_env = Environment::with_limits(Limits {
        cpu: 1_000_000,
        memory: 16 * 1024 * 1024,
        disk: 16 * 1024 * 1024,
        output: 8,
    });
    let (outcome, _, stderr) = diagnostic_env.run_script_capture("cat /work/missing");
    assert_eq!(
        outcome.exit_status,
        137,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
}

#[test]
fn independent_shell_sessions_share_files_but_keep_shell_state() {
    let mut env = Environment::new();
    let first = env.spawn_shell_session().unwrap();
    let second = env.spawn_shell_session().unwrap();
    assert_ne!(first, second);
    assert_eq!(env.scheduler.current(), Some(1_234));

    let first_run = env
        .run_shell_session_capture(
            first,
            "cd /work; export OWNER=first; printf shared > note; printf '%s:%s' \"$$\" \"$OWNER\"",
        )
        .unwrap();
    assert_eq!(first_run.0.exit_status, 0);
    assert_eq!(
        String::from_utf8_lossy(&first_run.1),
        format!("{first}:first")
    );

    let second_run = env
        .run_shell_session_capture(
            second,
            "cd /tmp; export OWNER=second; printf '%s:%s' \"$$\" \"$OWNER\"; cat /work/note",
        )
        .unwrap();
    assert_eq!(second_run.0.exit_status, 0);
    assert_eq!(
        String::from_utf8_lossy(&second_run.1),
        format!("{second}:secondshared")
    );

    assert_eq!(
        env.run_shell_session_capture(first, "printf '%s:%s' \"$PWD\" \"$OWNER\"")
            .unwrap()
            .1,
        b"/work:first"
    );
    assert_eq!(env.scheduler.current(), Some(1_234));
    assert_eq!(
        run(&mut env, "printf '%s:%s' \"$PWD\" \"${OWNER-unset}\"").1,
        "/:unset"
    );
}

#[test]
fn shell_directory_changes_use_process_scoped_system_state() {
    let mut env = Environment::new();
    let first = env.spawn_shell_session().unwrap();
    let second = env.spawn_shell_session().unwrap();

    assert_eq!(
        env.run_shell_session_capture(first, "cd /work; umask 077; pwd; umask")
            .unwrap()
            .1,
        b"/work\n0077\n"
    );
    assert_eq!(
        env.run_shell_session_capture(second, "pwd; umask")
            .unwrap()
            .1,
        b"/\n0022\n"
    );
    assert_eq!(
        env.run_shell_session_capture(first, "cd /missing; pwd")
            .unwrap()
            .1,
        b"/work\n"
    );
}

#[test]
fn exited_shell_session_cannot_accept_another_action() {
    let mut env = Environment::new();
    let session = env.spawn_shell_session().unwrap();
    let result = env.run_shell_session_capture(session, "exit 7").unwrap();
    assert_eq!(result.0.exit_status, 7);
    assert!(env
        .run_shell_session_capture(session, "printf stale")
        .is_err());
    assert_eq!(run(&mut env, "printf root").1, "root");
}

#[test]
fn idle_shell_sessions_survive_environment_snapshots() {
    let mut env = Environment::new();
    let session = env.spawn_shell_session().unwrap();
    env.run_shell_session_capture(session, "cd /work; export LABEL=kept")
        .unwrap();
    let mut fork = env.clone();
    assert_eq!(
        env.run_shell_session_capture(session, "printf '%s:%s' \"$PWD\" \"$LABEL\"")
            .unwrap()
            .1,
        b"/work:kept"
    );
    assert_eq!(
        fork.run_shell_session_capture(session, "printf '%s:%s' \"$PWD\" \"$LABEL\"")
            .unwrap()
            .1,
        b"/work:kept"
    );
    fork.run_shell_session_capture(session, "export LABEL=changed")
        .unwrap();
    assert_eq!(
        env.run_shell_session_capture(session, "printf '%s' \"$LABEL\"")
            .unwrap()
            .1,
        b"kept"
    );
}

#[test]
fn native_images_follow_executable_permissions_and_can_be_replaced() {
    let mut env = Environment::new();
    env.vfs.chmod("/", "/usr/bin/pwd", 0o644).unwrap();
    let denied = run(&mut env, "env pwd");
    assert_eq!(denied.0, 126);
    assert!(denied.2.contains("permission denied"), "{}", denied.2);
    assert_eq!(run(&mut env, "pwd"), (0, "/\n".into(), "".into()));
    assert_eq!(run(&mut env, "env -i PATH=/missing pwd").0, 127);

    env.vfs.remove_file("/", "/usr/bin/pwd").unwrap();
    let missing = run(&mut env, "env pwd");
    assert_eq!(missing.0, 127);
    assert!(missing.2.contains("command not found"), "{}", missing.2);
    assert_eq!(run(&mut env, "/usr/bin/pwd").0, 127);
    assert_eq!(run(&mut env, "which pwd").0, 1);

    env.vfs
        .write(
            "/",
            "/usr/bin/pwd",
            b"#!/bin/sh\nprintf 'replacement\\n'\n",
            0o755,
        )
        .unwrap();
    assert_eq!(
        run(&mut env, "env pwd"),
        (0, "replacement\n".into(), "".into())
    );
}

#[test]
fn proc_reports_a_parent_blocked_while_its_child_runs() {
    let mut env = Environment::new();
    let status = run(&mut env, "(cat /proc/1234/status)");
    assert_eq!(status.0, 0, "{}", status.2);
    assert!(status.1.contains("State:\tS (sleeping)"), "{}", status.1);
    assert_eq!(env.scheduler.current(), Some(1_234));
    assert_eq!(env.scheduler.state(1_234), Some(TaskState::Running));
}

#[test]
fn proc_environment_is_exported_sorted_and_nul_delimited() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "LOCAL=hidden; export PUBLIC=shown").0, 0);
    let environment = env.fs_read("/", "/proc/self/environ").unwrap();
    let entries = environment
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    assert!(entries.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(entries.iter().any(|entry| *entry == b"PUBLIC=shown"));
    assert!(!entries.iter().any(|entry| entry.starts_with(b"LOCAL=")));
}

#[test]
fn background_process_runs_on_a_later_scheduler_turn_and_wait_reaps_it() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "false &").0, 0);
    let status = env.fs_read("/", "/proc/1235/status").unwrap();
    let status = String::from_utf8(status).unwrap();
    assert!(status.contains("State:\tR (running)"), "{status}");
    assert!(status.contains("NSpgid:\t1235\n"), "{status}");
    assert!(status.contains("NSsid:\t1234\n"), "{status}");
    assert_eq!(env.scheduler.state(1_235), Some(TaskState::Runnable));
    assert_eq!(run(&mut env, "jobs").1, "[1] Running false\n");
    for _ in 0..8 {
        if matches!(env.scheduler.state(1_235), Some(TaskState::Exited(1))) {
            break;
        }
        run(&mut env, "true");
    }
    assert_eq!(run(&mut env, "jobs").1, "[1] Done false\n");

    let status = env.fs_read("/", "/proc/1235/status").unwrap();
    let status = String::from_utf8(status).unwrap();
    assert!(status.contains("State:\tZ (zombie)"), "{status}");
    assert!(status.contains("ExitCode:\t1"), "{status}");
    assert_eq!(env.scheduler.state(1_235), Some(TaskState::Exited(1)));
    assert_eq!(env.scheduler.current(), Some(1_234));
    assert!(run(&mut env, "ls /proc/1235/fd").1.trim().is_empty());
    assert_eq!(run(&mut env, "wait 1235").0, 1);
    assert!(env.fs_read("/", "/proc/1235/status").is_err());
    assert_eq!(env.scheduler.state(1_235), None);
}

#[test]
fn detached_output_is_delivered_once_when_the_child_later_exits() {
    let mut env = Environment::new();
    let (first, stdout, stderr) = env.run_script_capture("printf delayed &");
    assert_eq!(first.exit_status, 0);
    assert!(first.usage.memory_current > 0);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());

    let (second, stdout, stderr) = env.run_script_capture("printf foreground");
    assert_eq!(second.exit_status, 0);
    assert_eq!(stdout, b"delayedforeground");
    assert!(stderr.is_empty());
    assert_eq!(second.usage.memory_current, 0);

    assert_eq!(run(&mut env, "true").1, "");
}

#[test]
fn pipeline_backpressure_moves_more_than_one_pipe_capacity() {
    let mut env = Environment::new();
    assert_eq!(
        run(&mut env, "seq 20000 | wc -l"),
        (0, "20000\n".into(), "".into())
    );
}

#[test]
fn pseudo_files_are_read_only_and_do_not_consume_vfs_disk() {
    let mut env = Environment::new();
    let disk_before = env.vfs.disk_used();
    assert_eq!(run(&mut env, "cat /dev/null").0, 0);
    assert_eq!(
        run(
            &mut env,
            "printf input | cat /dev/stdin; printf out > /dev/stdout; printf err > /dev/stderr"
        ),
        (0, "inputout".into(), "err".into())
    );
    let write = run(&mut env, "printf no > /proc/created");
    assert_ne!(write.0, 0);
    assert!(write.2.contains("Read-only filesystem"), "{}", write.2);
    assert_eq!(env.vfs.disk_used(), disk_before);
}

#[test]
fn generated_devices_are_listed_and_stream_through_pipes_and_redirects() {
    let mut env = Environment::new();
    let listing = run(&mut env, "ls /dev");
    assert_eq!(listing.0, 0, "{}", listing.2);
    for name in ["null", "random", "urandom", "zero"] {
        assert!(
            listing.1.split_whitespace().any(|entry| entry == name),
            "{}",
            listing.1
        );
    }

    let (outcome, stdout, stderr) =
        env.run_script_capture("cat /dev/zero | head -c 32; head -c 8 < /dev/zero");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, vec![0; 40]);
    assert!(stderr.is_empty());
    assert_eq!(run(&mut env, "printf discarded > /dev/urandom").0, 0);
}

#[test]
fn simulated_random_devices_are_deterministic_and_distinct() {
    fn sample(path: &str) -> Vec<u8> {
        let mut env = Environment::new();
        let (outcome, stdout, stderr) = env.run_script_capture(&format!("head -c 64 {path}"));
        assert_eq!(
            outcome.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
        stdout
    }

    let random = sample("/dev/random");
    let urandom = sample("/dev/urandom");
    assert_eq!(random, sample("/dev/random"));
    assert_eq!(urandom, sample("/dev/urandom"));
    assert_ne!(random, urandom);
    assert!(random.iter().any(|byte| *byte != 0));

    for path in ["/dev/random", "/dev/urandom"] {
        let mut env = Environment::new();
        let (outcome, stdout, stderr) = env.run_script_capture(&format!("cat {path} | head"));
        assert_eq!(
            outcome.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
        assert_eq!(stdout.iter().filter(|byte| **byte == b'\n').count(), 10);
    }
}

#[test]
fn unterminated_infinite_device_stream_exhausts_cpu_fuel() {
    let mut env = Environment::with_limits(Limits {
        cpu: 5_000,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) = env.run_script_capture("cat /dev/zero | head");

    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert!(stdout.iter().all(|byte| *byte == 0));
}

#[test]
fn process_fork_is_rejected_before_copying_unbounded_shell_state() {
    let mut env = Environment::with_limits(Limits {
        memory: 4 * 1024,
        ..Limits::unlimited()
    });
    env.set_var("LARGE", "x".repeat(8 * 1024));
    let (outcome, _, _) = env.run_script_capture("(true)");
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    assert!(env.processes.get(1_235).is_none());
}

#[test]
fn kill_queues_signals_for_jobs_and_supports_existence_probes() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & pid=$!; kill -0 $pid; kill -INT %1; wait $pid; echo status:$?; kill -l 130",
    );
    assert_eq!(result, (0, "status:130\nINT\n".into(), String::new()));
    assert!(env.processes.get(1_235).is_none());
}

#[test]
fn job_signals_reach_the_background_process_group() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "(sleep 10) & sleep 1; kill -0 -- -$!; kill -TERM %1; wait %1; echo status:$?",
    );
    assert_eq!(result, (0, "status:143\n".into(), String::new()));
    let retained = env.processes.iter().cloned().collect::<Vec<_>>();
    assert!(
        retained.iter().all(|process| process.pid == 1_234),
        "{retained:?}"
    );
}

#[test]
fn terminal_foreground_group_returns_to_shell_after_job_exit() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "sleep 10 &").0, 0);
    env.terminal.set_foreground(&env.processes, 1_235).unwrap();
    assert_eq!(env.terminal.foreground_group, 1_235);
    assert_eq!(run(&mut env, "kill -TERM -- -1235; wait %1").0, 143);
    assert_eq!(env.terminal.foreground_group, 1_234);
}

#[test]
fn fg_uses_resumable_child_wait_and_restores_terminal() {
    let mut env = Environment::new();
    let result = run(&mut env, "sleep 2 & fg %1; printf 'status:%s' $?");
    assert_eq!(result, (0, "status:0".into(), String::new()));
    assert_eq!(env.clock.monotonic_ns(), 2_000_000_000);
    assert_eq!(env.terminal.foreground_group, 1_234);
    assert_eq!(run(&mut env, "fg").0, 1);
}

#[test]
fn stopped_jobs_are_observable_and_resume_through_bg() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & kill -STOP %1; jobs; bg %1; wait %1; printf 'status:%s' $?",
    );
    assert_eq!(
        result,
        (
            0,
            "[1] Stopped sleep 10\n[1] sleep 10 &\nstatus:0".into(),
            String::new(),
        )
    );
    assert_eq!(env.clock.monotonic_ns(), 10_000_000_000);
    assert_eq!(env.terminal.foreground_group, 1_234);
}

#[test]
fn timer_wake_is_retained_while_a_job_is_stopped() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & sleep 1; kill -STOP %1; sleep 20; bg %1; wait %1; printf done",
    );
    assert_eq!(result, (0, "[1] sleep 10 &\ndone".into(), String::new()));
    assert_eq!(env.clock.monotonic_ns(), 21_000_000_000);
}

#[test]
fn terminating_signal_stays_pending_until_a_stopped_job_continues() {
    let mut env = Environment::new();
    let result = run(
        &mut env,
        "sleep 10 & sleep 1; kill -STOP %1; kill -TERM %1; kill -CONT %1; wait %1",
    );
    assert_eq!(result, (143, String::new(), String::new()));
    assert_eq!(env.clock.monotonic_ns(), 1_000_000_000);
}

#[test]
fn kill_reaches_a_stopped_job_and_proc_reports_the_state() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "sleep 10 & kill -STOP %1").0, 0);
    let status = run(&mut env, "cat /proc/1235/status; ps aux");
    assert_eq!(status.0, 0);
    assert!(status.1.contains("State:\tT (stopped)\n"), "{}", status.1);
    assert!(
        status.1.contains(" T    00:00   0:00 sleep 10"),
        "{}",
        status.1
    );
    assert!(status.2.is_empty());
    assert_eq!(run(&mut env, "kill -KILL %1; wait %1").0, 137);
}

#[test]
fn a_signal_to_the_persistent_shell_terminates_the_session() {
    let mut env = Environment::new();
    let result = run(&mut env, "kill -HUP $$; echo unreachable");
    assert_eq!(result, (129, String::new(), String::new()));
    assert!(env.is_terminated());
    assert_eq!(env.termination_status(), Some(129));
}

#[test]
fn trap_dispositions_are_queryable_atomic_and_keep_kill_uncatchable() {
    let mut env = Environment::new();
    let result = run(&mut env, "trap 'printf caught' TERM; trap '' HUP; trap -p");
    assert_eq!(
        result,
        (
            0,
            "trap -- '' SIGHUP\ntrap -- 'printf caught' SIGTERM\n".into(),
            String::new(),
        )
    );

    let result = run(&mut env, "trap 'printf bad' INT KILL; trap -p INT");
    assert_eq!(result.0, 0);
    assert!(result.1.is_empty());
    assert!(
        result.2.contains("SIGKILL cannot be caught"),
        "{}",
        result.2
    );
}

#[test]
fn child_signal_traps_and_exec_disposition_rules_use_process_state() {
    let mut env = Environment::new();
    assert_eq!(
        run(
            &mut env,
            "trap 'printf child' CHLD; sleep 1 & wait; printf done",
        ),
        (0, "childdone".into(), String::new())
    );

    let mut env = Environment::new();
    assert_eq!(
        run(
            &mut env,
            "trap 'printf inherited' TERM; bash -c 'kill -TERM $$; printf unreachable'; printf status:$?",
        ),
        (0, "status:143".into(), String::new())
    );

    let mut env = Environment::new();
    assert_eq!(
        run(
            &mut env,
            "trap '' TERM; bash -c 'kill -TERM $$; printf survived'",
        ),
        (0, "survived".into(), String::new())
    );
}

#[test]
fn trap_state_growth_is_bounded_before_installation() {
    let mut env = Environment::new();
    let source = format!("trap '{}' TERM", "x".repeat(1024 * 1024));
    let result = run(&mut env, &source);
    assert_eq!(result.0, 2);
    assert!(result.2.contains("handler state exceeds"), "{}", result.2);
}

#[test]
fn subshell_execs_its_final_native_command_in_place() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run(
        &mut env,
        "sleep 10 & echo bg=$!; ps -o pid,ppid,args | cat; kill $!",
    );
    assert_eq!(status, 0, "{stderr}");
    let lines = stdout.lines().collect::<Vec<_>>();
    assert_eq!(lines[0], "bg=1235");
    // `$!` is the sleep process itself, and the `ps` pipeline stage is a direct shell child.
    assert!(lines.contains(&"1235 1234 sleep 10"), "{stdout}");
    assert!(
        lines
            .iter()
            .any(|line| line.ends_with(" 1234 ps -o pid,ppid,args")),
        "{stdout}"
    );
    assert!(
        !lines.iter().any(|line| line.contains(" 1235 ")),
        "background job spawned a grandchild: {stdout}"
    );
}

#[test]
fn builtin_writer_to_a_closed_pipe_dies_of_sigpipe() {
    let mut env = Environment::new();
    assert_eq!(
        run(
            &mut env,
            "set -o pipefail; { sleep 1; printf x; } | /usr/bin/true; echo $?",
        ),
        (0, "141\n".into(), String::new())
    );
    let (status, stdout, stderr) = run(
        &mut env,
        "trap '' PIPE; { sleep 1; printf x; } | /usr/bin/true; echo $?",
    );
    assert_eq!((status, stdout.as_str()), (0, "1\n"));
    assert!(stderr.contains("broken pipe"), "{stderr}");
}

#[test]
fn pipeline_stage_with_long_command_text_runs() {
    let mut env = Environment::new();
    let literal = "a".repeat(10_000);
    let (status, stdout, stderr) = run(&mut env, &format!("printf %s {literal} | wc -c"));
    assert_eq!(status, 0, "{}", &stderr[..stderr.len().min(200)]);
    assert_eq!(stdout, "10000\n");
}

#[test]
fn shell_reads_its_program_from_stdin_one_command_at_a_time() {
    let mut env = Environment::new();
    // Each command runs before the next line is read, so `read` consumes the following line,
    // `exit` leaves the rest unread, and an endless producer is consumed incrementally.
    let (status, stdout, stderr) = run(
        &mut env,
        "printf 'read x\\nhello\\necho \"got $x\"\\n' | bash; \
         printf 'echo one\\nexit 3\\necho never\\n' | sh; echo s=$?; \
         yes 'echo y' | sh | head -n 2; \
         printf 'if true; then\\n echo multi\\nfi\\n' | sh",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "got hello\none\ns=3\ny\ny\nmulti\n");
}

#[test]
fn shell_image_options_set_parameters_and_drop_unexported_state() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run(
        &mut env,
        "f() { :; }; A=1; export B=2; \
         bash -c 'echo \"$0 a=$A b=$B\"; type f >/dev/null 2>&1 || echo nofn'; \
         sh -c 'echo $0 $1 $#' name one two; \
         echo 'echo \"$0 [$*]\"' | bash -s p q; \
         printf 'echo \"$0:$1\"\\n' > /script.sh; bash /script.sh arg; \
         bash -o pipefail -c 'false | true; echo pf=$?'; \
         bash -e -c 'false; echo unreachable'; echo e=$?",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        "bash a= b=2\nnofn\nname one 2\nbash [p q]\n/script.sh:arg\npf=1\ne=1\n"
    );
}

#[test]
fn shell_image_rejects_invalid_invocations_and_oversized_stdin_commands() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run(
        &mut env,
        "bash -z -c true; echo opt=$?; bash /missing; echo missing=$?; \
         bash -c 'if'; echo syntax=$?; echo 'echo \"a' | sh; echo quote=$?; \
         head -c 300000 /dev/zero | tr '\\0' a | sh; echo long=$?",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "opt=2\nmissing=127\nsyntax=2\nquote=2\nlong=2\n");
    assert!(stderr.contains("bash: -z: invalid option"), "{stderr}");
    assert!(stderr.contains("bash: /missing: No such file"), "{stderr}");
    assert!(
        stderr.contains("standard input command too long"),
        "{stderr}"
    );
}
