//! Process-level syscall sandbox.
//!
//! The command layer never intentionally executes host programs or opens host network sockets.
//! This module adds a seccomp backstop so a future bug or dependency cannot silently broaden
//! that capability boundary. It denies network egress, native process execution/fork, and
//! `ptrace` before any simulated program runs.
//!
//! This is safe for shellsim because the simulator never opens sockets, execs programs, or
//! forks — it is entirely in-process with a virtual clock and virtual network. The filter is
//! **default-allow**: only the explicit denylist is blocked. It remains defense-in-depth, not a
//! complete host-filesystem confinement mechanism.
//!
//! Set `SHELLSIM_NO_SANDBOX=1` to skip it (debugging / unusual hosts).

use std::process::Command;

/// Install the syscall sandbox. No-op on non-Linux and when `SHELLSIM_NO_SANDBOX` is set.
pub fn apply() {
    if std::env::var_os("SHELLSIM_NO_SANDBOX").is_some() {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        if let Err(e) = imp::install() {
            // Don't hard-fail: a host that forbids seccomp (some CI/containers) should still
            // run, just without the kernel backstop.
            eprintln!("shellsim: warning: seccomp sandbox not installed ({e}); continuing without the kernel backstop");
        }
    }
}

/// Arrange for a child process to retain host reads but reject filesystem mutation.
///
/// The filter is installed after `fork` and before `exec`, so it affects only the child. It also
/// denies network access and further process creation. The caller must use pipes or `/dev/null`
/// for standard output and error because writes to those two descriptors remain available. This
/// is Linux-only defense in depth; other platforms retain the caller's ordinary permissions.
pub(crate) fn constrain_child_to_read_only(command: &mut Command) -> Result<(), String> {
    if std::env::var_os("SHELLSIM_NO_SANDBOX").is_some() {
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        imp::constrain_child_to_read_only(command)
            .map_err(|error| format!("cannot build read-only child sandbox: {error}"))?;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod imp {
    use seccompiler::{
        BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter,
        SeccompRule,
    };
    use std::collections::BTreeMap;
    use std::io;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    pub fn install() -> Result<(), Box<dyn std::error::Error>> {
        // Syscalls present on every Linux arch we target. Blocking `socket`/`connect` kills all
        // network egress (you cannot connect without first creating a socket); blocking
        // `execve`/`execveat` means even a successful fork/posix_spawn cannot run a real program,
        // which neuters `os.system`/`subprocess`. `ptrace` is blocked so a child cannot be used
        // to poke another process.
        let mut denied: Vec<i64> = vec![
            libc::SYS_socket,
            libc::SYS_connect,
            libc::SYS_socketpair,
            libc::SYS_execve,
            libc::SYS_execveat,
            libc::SYS_ptrace,
        ];
        // fork/vfork don't exist as distinct syscalls on every arch (aarch64 routes through
        // clone); include them where the libc constant is defined. Denying exec already blocks
        // running new programs — this is belt-and-suspenders for raw process creation.
        #[cfg(target_arch = "x86_64")]
        {
            denied.push(libc::SYS_fork);
            denied.push(libc::SYS_vfork);
        }

        let mut rules: BTreeMap<i64, Vec<seccompiler::SeccompRule>> = BTreeMap::new();
        for s in denied {
            rules.insert(s, vec![]); // empty rule vec == match the syscall unconditionally
        }

        let arch = std::env::consts::ARCH.try_into()?;
        let filter = SeccompFilter::new(
            rules,
            SeccompAction::Allow, // default for everything not listed
            SeccompAction::Errno(libc::EPERM as u32), // listed syscalls fail with EPERM
            arch,
        )?;
        let prog: BpfProgram = filter.try_into()?;
        seccompiler::apply_filter(&prog)?;
        Ok(())
    }

    pub fn constrain_child_to_read_only(
        command: &mut Command,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let program = read_only_program()?;
        // SAFETY: the closure only installs a precompiled BPF program through `prctl` and
        // `seccomp`. It neither touches shared process state nor allocates on the success path.
        unsafe {
            command.pre_exec(move || {
                seccompiler::apply_filter(&program)
                    .map_err(|_| io::Error::from_raw_os_error(libc::EPERM))
            });
        }
        Ok(())
    }

    fn read_only_program() -> Result<BpfProgram, Box<dyn std::error::Error>> {
        let mut rules = BTreeMap::new();
        rules.insert(libc::SYS_openat, mutating_open_rules(2)?);
        #[cfg(target_arch = "x86_64")]
        rules.insert(libc::SYS_open, mutating_open_rules(1)?);
        rules.insert(libc::SYS_write, non_output_write_rules()?);
        rules.insert(libc::SYS_writev, non_output_write_rules()?);
        rules.insert(
            libc::SYS_mmap,
            vec![SeccompRule::new(vec![
                SeccompCondition::new(
                    2,
                    SeccompCmpArgLen::Dword,
                    SeccompCmpOp::MaskedEq(libc::PROT_WRITE as u64),
                    libc::PROT_WRITE as u64,
                )?,
                SeccompCondition::new(
                    3,
                    SeccompCmpArgLen::Dword,
                    SeccompCmpOp::MaskedEq(libc::MAP_SHARED as u64),
                    libc::MAP_SHARED as u64,
                )?,
            ])?],
        );

        let mut denied = vec![
            libc::SYS_openat2,
            libc::SYS_renameat,
            libc::SYS_renameat2,
            libc::SYS_unlinkat,
            libc::SYS_mkdirat,
            libc::SYS_mknodat,
            libc::SYS_linkat,
            libc::SYS_symlinkat,
            libc::SYS_fchmod,
            libc::SYS_fchmodat,
            libc::SYS_fchown,
            libc::SYS_fchownat,
            libc::SYS_truncate,
            libc::SYS_ftruncate,
            libc::SYS_fallocate,
            libc::SYS_pwrite64,
            libc::SYS_pwritev,
            libc::SYS_pwritev2,
            libc::SYS_copy_file_range,
            libc::SYS_sendfile,
            libc::SYS_splice,
            libc::SYS_utimensat,
            libc::SYS_setxattr,
            libc::SYS_lsetxattr,
            libc::SYS_fsetxattr,
            libc::SYS_removexattr,
            libc::SYS_lremovexattr,
            libc::SYS_fremovexattr,
            libc::SYS_mount,
            libc::SYS_umount2,
            libc::SYS_pivot_root,
            libc::SYS_move_mount,
            libc::SYS_fsopen,
            libc::SYS_fsconfig,
            libc::SYS_fsmount,
            libc::SYS_mount_setattr,
            libc::SYS_io_uring_setup,
            libc::SYS_io_uring_enter,
            libc::SYS_io_uring_register,
            libc::SYS_socket,
            libc::SYS_connect,
            libc::SYS_socketpair,
            libc::SYS_clone,
            libc::SYS_clone3,
            libc::SYS_ptrace,
        ];
        // `fchmodat2` has the same number on every architecture supported by seccompiler, but the
        // libc bindings do not expose its symbolic constant on every one of them yet.
        denied.push(452);
        #[cfg(target_arch = "x86_64")]
        denied.extend([
            libc::SYS_creat,
            libc::SYS_rename,
            libc::SYS_unlink,
            libc::SYS_mkdir,
            libc::SYS_rmdir,
            libc::SYS_link,
            libc::SYS_symlink,
            libc::SYS_mknod,
            libc::SYS_chmod,
            libc::SYS_chown,
            libc::SYS_lchown,
            libc::SYS_utime,
            libc::SYS_fork,
            libc::SYS_vfork,
        ]);
        for syscall in denied {
            rules.insert(syscall, vec![]);
        }

        let arch = std::env::consts::ARCH.try_into()?;
        Ok(SeccompFilter::new(
            rules,
            SeccompAction::Allow,
            SeccompAction::Errno(libc::EPERM as u32),
            arch,
        )?
        .try_into()?)
    }

    fn mutating_open_rules(argument: u8) -> Result<Vec<SeccompRule>, Box<dyn std::error::Error>> {
        [
            libc::O_CREAT,
            libc::O_TRUNC,
            libc::O_APPEND,
            libc::O_TMPFILE,
        ]
        .into_iter()
        .map(|flag| {
            Ok(SeccompRule::new(vec![SeccompCondition::new(
                argument,
                SeccompCmpArgLen::Dword,
                SeccompCmpOp::MaskedEq(flag as u64),
                flag as u64,
            )?])?)
        })
        .collect()
    }

    fn non_output_write_rules() -> Result<Vec<SeccompRule>, Box<dyn std::error::Error>> {
        [
            SeccompCondition::new(0, SeccompCmpArgLen::Dword, SeccompCmpOp::Eq, 0)?,
            SeccompCondition::new(0, SeccompCmpArgLen::Dword, SeccompCmpOp::Ge, 3)?,
        ]
        .into_iter()
        .map(|condition| Ok(SeccompRule::new(vec![condition])?))
        .collect()
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::process::Command;

    #[test]
    fn read_only_child_can_read_but_cannot_create_a_file() {
        let directory = std::env::temp_dir().join(format!(
            "shellsim-read-only-child-test-{}",
            std::process::id()
        ));
        let input = directory.join("input");
        let output = directory.join("output");
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(&input, "readable\n").unwrap();

        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "IFS= read -r value < \"$1\"; printf %s \"$value\"; printf blocked > \"$2\"",
                "shell",
            ])
            .arg(&input)
            .arg(&output);
        super::imp::constrain_child_to_read_only(&mut command).unwrap();
        let result = command.output().unwrap();

        assert!(!result.status.success());
        assert_eq!(
            result.stdout,
            b"readable",
            "status {}; {}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!output.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
