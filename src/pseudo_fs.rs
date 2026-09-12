//! Read-only synthetic filesystem views for modeled process and device state.
//!
//! These nodes are generated from `Environment` and never materialized in the VFS. They therefore
//! cannot expose host `/proc`, consume simulated disk, or become stale snapshots. The interface is
//! intentionally eager only for finite files; streaming devices such as `/dev/zero` remain outside
//! the supported frontier.

use crate::interp::Environment;
use crate::process::ProcessStatus;
use crate::scheduler::TaskState;
use crate::vfs::{resolve_against, Node, NodeKind, VfsError};

enum PseudoNode {
    File(Vec<u8>),
    Directory(Vec<String>),
    Symlink(String),
    Error(VfsError),
}

const MAX_PSEUDO_FILE_BYTES: usize = 4 * 1024 * 1024;

fn absolute_path(env: &Environment, cwd: &str, path: &str, follow_self: bool) -> String {
    let path = resolve_against(cwd, path);
    if path == "/proc/self" && follow_self {
        return format!("/proc/{}", env.pid);
    }
    if let Some(rest) = path.strip_prefix("/proc/self/") {
        return format!("/proc/{}/{rest}", env.pid);
    }
    path
}

fn lookup(env: &Environment, cwd: &str, path: &str, follow_self: bool) -> Option<PseudoNode> {
    let path = absolute_path(env, cwd, path, follow_self);
    match path.as_str() {
        "/dev" => {
            return Some(PseudoNode::Directory(vec![
                "null".to_string(),
                "stderr".to_string(),
                "stdin".to_string(),
                "stdout".to_string(),
            ]));
        }
        "/dev/null" => return Some(PseudoNode::File(Vec::new())),
        "/dev/stdin" => return Some(PseudoNode::Symlink("/proc/self/fd/0".to_string())),
        "/dev/stdout" => return Some(PseudoNode::Symlink("/proc/self/fd/1".to_string())),
        "/dev/stderr" => return Some(PseudoNode::Symlink("/proc/self/fd/2".to_string())),
        "/proc/self" => return Some(PseudoNode::Symlink(env.pid.to_string())),
        "/proc" => {
            let mut entries = vec![
                "cpuinfo".to_string(),
                "meminfo".to_string(),
                "mounts".to_string(),
                "self".to_string(),
                "uptime".to_string(),
                "version".to_string(),
            ];
            entries.extend(env.processes.iter().map(|process| process.pid.to_string()));
            entries.sort();
            return Some(PseudoNode::Directory(entries));
        }
        "/proc/uptime" => {
            let seconds = env.clock.monotonic_seconds();
            return Some(PseudoNode::File(
                format!("{seconds:.2} {seconds:.2}\n").into_bytes(),
            ));
        }
        "/proc/meminfo" => {
            let limit = env.resources.limits().memory;
            let used = env.resources.memory_mark();
            return Some(PseudoNode::File(
                format!(
                    "MemTotal:       {} kB\nMemFree:        {} kB\nMemAvailable:   {} kB\nSwapTotal:      0 kB\nSwapFree:       0 kB\n",
                    limit / 1024,
                    limit.saturating_sub(used) / 1024,
                    limit.saturating_sub(used) / 1024,
                )
                .into_bytes(),
            ));
        }
        "/proc/cpuinfo" => {
            return Some(PseudoNode::File(
                b"processor\t: 0\nmodel name\t: shellsim virtual CPU\n".to_vec(),
            ));
        }
        "/proc/version" => {
            return Some(PseudoNode::File(
                b"Linux version 6.12.0-shellsim (deterministic simulator)\n".to_vec(),
            ));
        }
        "/proc/mounts" => {
            return Some(PseudoNode::File(
                b"shellsim / rootfs rw 0 0\nproc /proc proc ro 0 0\ndev /dev devtmpfs ro 0 0\n"
                    .to_vec(),
            ));
        }
        _ => {}
    }

    let rest = path.strip_prefix("/proc/")?;
    let (pid_text, leaf) = rest.split_once('/').unwrap_or((rest, ""));
    let pid = pid_text.parse().ok()?;
    let process = env.processes.get(pid)?;
    let cwd = if pid == env.pid {
        env.cwd.as_str()
    } else {
        process.cwd.as_str()
    };
    match leaf {
        "" => Some(PseudoNode::Directory(vec![
            "cmdline".to_string(),
            "cwd".to_string(),
            "environ".to_string(),
            "status".to_string(),
        ])),
        "cwd" => Some(PseudoNode::Symlink(cwd.to_string())),
        "cmdline" => {
            let mut value = process.command.as_bytes().to_vec();
            value.push(0);
            Some(PseudoNode::File(value))
        }
        "environ" => {
            let mut value = Vec::new();
            let entries: Box<dyn Iterator<Item = (&String, &String)> + '_> = if pid == env.pid {
                Box::new(
                    env.exported
                        .iter()
                        .filter_map(|name| env.vars.get_key_value(name)),
                )
            } else {
                Box::new(process.environment.iter())
            };
            for (name, contents) in entries {
                let required = name
                    .len()
                    .checked_add(contents.len())
                    .and_then(|size| size.checked_add(2));
                if required.is_none_or(|required| {
                    value.len().saturating_add(required) > MAX_PSEUDO_FILE_BYTES
                }) {
                    return Some(PseudoNode::Error(VfsError::TooLarge {
                        path,
                        limit: MAX_PSEUDO_FILE_BYTES,
                    }));
                }
                value.extend_from_slice(name.as_bytes());
                value.push(b'=');
                value.extend_from_slice(contents.as_bytes());
                value.push(0);
            }
            Some(PseudoNode::File(value))
        }
        "status" => {
            let task_state = env.scheduler.state(pid);
            let (state, status) = match (process.status, task_state) {
                (ProcessStatus::Exited(status), _) | (_, Some(TaskState::Exited(status))) => {
                    ("Z (zombie)", Some(status))
                }
                (_, Some(TaskState::Blocked(_))) => ("S (sleeping)", None),
                (_, Some(TaskState::Runnable | TaskState::Running))
                | (ProcessStatus::Running, None) => ("R (running)", None),
            };
            let exit = status.map_or(String::new(), |status| format!("ExitCode:\t{status}\n"));
            Some(PseudoNode::File(
                format!(
                    "Name:\t{}\nState:\t{state}\nPid:\t{}\nPPid:\t{}\nUid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0\n{exit}",
                    process.command, process.pid, process.ppid,
                )
                .into_bytes(),
            ))
        }
        _ => None,
    }
}

/// Read a finite pseudo-file, returning `None` when the path belongs to the ordinary VFS.
pub fn read(env: &Environment, cwd: &str, path: &str) -> Option<crate::vfs::Result<Vec<u8>>> {
    lookup(env, cwd, path, true).map(|node| match node {
        PseudoNode::File(data) => Ok(data),
        PseudoNode::Directory(_) => Err(VfsError::IsADir(path.to_string())),
        PseudoNode::Symlink(_) => Err(VfsError::NotFound(path.to_string())),
        PseudoNode::Error(error) => Err(error),
    })
}

/// Return generated metadata for a pseudo node.
pub fn metadata(
    env: &Environment,
    cwd: &str,
    path: &str,
    follow: bool,
) -> Option<crate::vfs::Result<Node>> {
    lookup(env, cwd, path, follow).map(|node| {
        let (kind, mode) = match node {
            PseudoNode::File(data) => (NodeKind::File(data), 0o444),
            PseudoNode::Directory(_) => (NodeKind::Dir, 0o555),
            PseudoNode::Symlink(target) => (NodeKind::Symlink(target), 0o777),
            PseudoNode::Error(error) => return Err(error),
        };
        Ok(Node {
            kind,
            mode,
            uid: 0,
            gid: 0,
            mtime: env.clock.unix_ms(),
        })
    })
}

/// List a generated pseudo-directory in deterministic order.
pub fn list_dir(
    env: &Environment,
    cwd: &str,
    path: &str,
) -> Option<crate::vfs::Result<Vec<String>>> {
    lookup(env, cwd, path, true).map(|node| match node {
        PseudoNode::Directory(entries) => Ok(entries),
        PseudoNode::Error(error) => Err(error),
        _ => Err(VfsError::NotADir(path.to_string())),
    })
}

/// Read a generated pseudo-symlink without following it.
pub fn read_link(env: &Environment, cwd: &str, path: &str) -> Option<crate::vfs::Result<String>> {
    lookup(env, cwd, path, false).map(|node| match node {
        PseudoNode::Symlink(target) => Ok(target),
        PseudoNode::Error(error) => Err(error),
        _ => Err(VfsError::Invalid(path.to_string())),
    })
}
