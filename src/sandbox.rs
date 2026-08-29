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

#[cfg(target_os = "linux")]
mod imp {
    use seccompiler::{BpfProgram, SeccompAction, SeccompFilter};
    use std::collections::BTreeMap;

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
}
