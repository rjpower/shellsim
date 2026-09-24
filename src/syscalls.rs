//! Process-owned filesystem operations shared by native execution and guest ABI adapters.
//!
//! Paths resolve only in the virtual filesystem. Open files live in the process descriptor
//! table, so a guest cannot keep a second, unaccounted set of file handles or cursors.

use crate::descriptors::{DescriptorError, Fd, FileState, IoPoll, MAX_FDS_PER_PROCESS};
use crate::display::{DisplayError, KeyEvent};
use crate::interp::Interp;
use crate::vfs::{resolve_against, NodeKind, VfsError};

/// Operations available to a native process during one scheduler quantum. The borrowed handle
/// cannot outlive the poll, so blocked programs retain only their own data and wait reason.
pub(crate) trait NativeSyscalls {
    fn cwd(&self) -> &str;
    fn write(&mut self, fd: Fd, bytes: &[u8]) -> Result<IoPoll<usize>, String>;
    fn charge_cpu(&mut self, units: u64) -> bool;
    fn output_remaining(&self) -> u64;
    fn charge_output(&mut self, bytes: u64) -> bool;
    fn stop_status(&self) -> i32;
}

/// Active-PID adapter; native program bodies receive this handle, not `Interp`.
/// The current WASI adapter still holds `Interp` while translating its imports.
pub(crate) struct ActiveProcessSyscalls<'a> {
    interp: &'a mut Interp,
}

impl<'a> ActiveProcessSyscalls<'a> {
    pub(crate) fn new(interp: &'a mut Interp) -> Self {
        Self { interp }
    }
}

impl NativeSyscalls for ActiveProcessSyscalls<'_> {
    fn cwd(&self) -> &str {
        &self.interp.process.cwd
    }

    fn write(&mut self, fd: Fd, bytes: &[u8]) -> Result<IoPoll<usize>, String> {
        self.interp.write_fd(fd, bytes)
    }

    fn charge_cpu(&mut self, units: u64) -> bool {
        self.interp.resources.charge_cpu(units)
    }

    fn output_remaining(&self) -> u64 {
        self.interp.resources.output_remaining()
    }

    fn charge_output(&mut self, bytes: u64) -> bool {
        self.interp.resources.charge_output(bytes)
    }

    fn stop_status(&self) -> i32 {
        self.interp
            .resources
            .stop_reason()
            .map_or(137, |reason| reason.exit_status())
    }
}

/// Open the single virtual display for the active process.
pub(crate) fn display_open(
    interp: &mut Interp,
    width: u32,
    height: u32,
    format: u32,
) -> Result<u32, DisplayError> {
    interp.display.open(
        &mut interp.resources,
        interp.process.pid,
        width,
        height,
        format,
    )
}

/// Copy a complete RGBA frame from a process into the virtual display.
pub(crate) fn display_present(
    interp: &mut Interp,
    handle: u32,
    pixels: &[u8],
    stride: u32,
) -> Result<(), DisplayError> {
    interp.display.present(
        &mut interp.resources,
        interp.process.pid,
        handle,
        pixels,
        stride,
    )
}

/// Receive one injected key event, or report that no event is ready.
pub(crate) fn input_poll_key(
    interp: &mut Interp,
    handle: u32,
) -> Result<Option<KeyEvent>, DisplayError> {
    interp
        .display
        .poll_key(&mut interp.resources, interp.process.pid, handle)
}

/// Relinquish the display while retaining its last frame for inspection.
pub(crate) fn display_close(interp: &mut Interp, handle: u32) -> Result<(), DisplayError> {
    interp.display.close(interp.process.pid, handle)
}

/// Access and creation requested by one virtual process.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OpenFile {
    pub readable: bool,
    pub writable: bool,
    pub create: bool,
    pub exclusive: bool,
    pub truncate: bool,
    pub append: bool,
}

/// Typed failure at the simulated process/kernel boundary.
#[derive(Debug)]
pub(crate) enum SyscallError {
    File(VfsError),
    Descriptor(DescriptorError),
    InvalidArgument,
    IsDirectory,
    Permission,
}

impl From<VfsError> for SyscallError {
    fn from(error: VfsError) -> Self {
        Self::File(error)
    }
}

impl From<DescriptorError> for SyscallError {
    fn from(error: DescriptorError) -> Self {
        Self::Descriptor(error)
    }
}

/// Open a regular virtual file in the active process, assigning the lowest free guest fd.
pub(crate) fn open_file(
    interp: &mut Interp,
    cwd: &str,
    path: &str,
    options: OpenFile,
) -> Result<Fd, SyscallError> {
    if path.is_empty() || path.contains('\0') || (!options.readable && !options.writable) {
        return Err(SyscallError::InvalidArgument);
    }
    if interp.process.fds.iter().count() >= MAX_FDS_PER_PROCESS {
        return Err(SyscallError::Descriptor(DescriptorError::DescriptorLimit));
    }
    // WASI adapters reserve 3 for the working directory and 4 for the VFS root.
    let fd = (5..MAX_FDS_PER_PROCESS as Fd + 5)
        .find(|&candidate| interp.process.fds.get(candidate).is_err())
        .ok_or(SyscallError::Descriptor(DescriptorError::DescriptorLimit))?;
    let absolute = resolve_against(cwd, path);
    match interp.vfs.metadata("/", &absolute, true) {
        Ok(node) => {
            if matches!(node.kind, NodeKind::Dir) {
                return Err(SyscallError::IsDirectory);
            }
            if options.create && options.exclusive {
                return Err(SyscallError::File(VfsError::Exists(absolute)));
            }
            if options.truncate {
                if !options.writable {
                    return Err(SyscallError::Permission);
                }
                interp.sync_vfs_time();
                interp.vfs.write("/", &absolute, &[], 0o666)?;
            }
        }
        Err(VfsError::NotFound(_)) if options.create && options.writable => {
            interp.sync_vfs_time();
            interp.vfs.write("/", &absolute, &[], 0o666)?;
        }
        Err(error) => return Err(error.into()),
    }
    let backing_path = interp.vfs.realpath(&absolute, true)?;
    let cursor = if options.append {
        interp.vfs.file_len("/", &backing_path)? as u64
    } else {
        0
    };
    let description = interp.descriptors.open_file(
        backing_path,
        cursor,
        options.readable,
        options.writable,
        false,
    )?;
    interp.install_new_description(fd, description)?;
    Ok(fd)
}

/// Change virtual permission bits for a path owned by the active process.
pub(crate) fn chmod(
    interp: &mut Interp,
    cwd: &str,
    path: &str,
    mode: u32,
) -> Result<(), SyscallError> {
    if path.is_empty() || path.contains('\0') || mode & !0o7777 != 0 {
        return Err(SyscallError::InvalidArgument);
    }
    interp.sync_vfs_time();
    interp.vfs.chmod(cwd, path, mode)?;
    Ok(())
}

/// Return regular-file state for an active process descriptor.
pub(crate) fn file_state(interp: &Interp, fd: Fd) -> Result<FileState, SyscallError> {
    let description = interp.process.fds.get(fd)?;
    interp
        .descriptors
        .file_state(description)?
        .ok_or(SyscallError::Descriptor(DescriptorError::WrongAccess))
}

/// Close a process-owned descriptor and update process metadata.
pub(crate) fn close(interp: &mut Interp, fd: Fd) -> Result<(), SyscallError> {
    interp.process.fds.close(fd, &mut interp.descriptors)?;
    interp
        .vfs
        .retain_orphans(&interp.descriptors.live_orphans());
    interp.refresh_descriptor_snapshot(interp.process.pid);
    Ok(())
}

/// Remove one regular file through the virtual filesystem, without exposing host paths.
pub(crate) fn unlink(interp: &mut Interp, cwd: &str, path: &str) -> Result<(), SyscallError> {
    interp.sync_vfs_time();
    let absolute = resolve_against(cwd, path);
    let entry = interp.vfs.realpath(&absolute, false)?;
    if interp.descriptors.has_open_file(&entry)
        && matches!(
            interp.vfs.metadata("/", &entry, false)?.kind,
            NodeKind::File(_)
        )
    {
        let orphan = interp.vfs.unlink_open_file("/", &entry)?;
        interp.descriptors.detach_file(&entry, orphan);
        Ok(())
    } else {
        interp.vfs.remove_file(cwd, path).map_err(Into::into)
    }
}

/// Create one directory; unlike `mkdir_all`, this retains POSIX's existing-parent requirement.
pub(crate) fn mkdir(interp: &mut Interp, cwd: &str, path: &str) -> Result<(), SyscallError> {
    interp.sync_vfs_time();
    interp.vfs.mkdir(cwd, path).map_err(Into::into)
}

/// Remove only an empty virtual directory.
pub(crate) fn rmdir(interp: &mut Interp, cwd: &str, path: &str) -> Result<(), SyscallError> {
    interp.sync_vfs_time();
    interp.vfs.rmdir(cwd, path).map_err(Into::into)
}

/// Move a virtual path within the same modeled filesystem.
pub(crate) fn rename(
    interp: &mut Interp,
    cwd: &str,
    from: &str,
    to: &str,
) -> Result<(), SyscallError> {
    interp.sync_vfs_time();
    interp.vfs.rename(cwd, from, to).map_err(Into::into)
}

/// Seek one regular-file description. The returned cursor is shared with duplicated fds.
pub(crate) fn seek(
    interp: &mut Interp,
    fd: Fd,
    delta: i64,
    whence: u32,
) -> Result<u64, SyscallError> {
    let description = interp.process.fds.get(fd)?;
    let file = file_state(interp, fd)?;
    let base = match whence {
        0 => 0_i128,
        1 => i128::from(file.cursor),
        2 => match file.orphan {
            Some(id) => interp.vfs.orphan_len(id)? as i128,
            None => interp.vfs.file_len("/", &file.path)? as i128,
        },
        _ => return Err(SyscallError::InvalidArgument),
    };
    let position =
        u64::try_from(base + i128::from(delta)).map_err(|_| SyscallError::InvalidArgument)?;
    interp.descriptors.seek_file(description, position)?;
    Ok(position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptors::IoPoll;

    #[test]
    fn opened_file_uses_process_descriptor_cursor_and_close() {
        let mut interp = Interp::new();
        interp
            .vfs
            .write("/", "/work/data", b"abcdef", 0o644)
            .unwrap();
        let fd = open_file(
            &mut interp,
            "/work",
            "data",
            OpenFile {
                readable: true,
                writable: false,
                create: false,
                exclusive: false,
                truncate: false,
                append: false,
            },
        )
        .unwrap();
        assert_eq!(
            interp.read_fd(fd, 2).unwrap(),
            IoPoll::Ready(b"ab".to_vec())
        );
        assert_eq!(file_state(&interp, fd).unwrap().cursor, 2);
        assert_eq!(seek(&mut interp, fd, -2, 2).unwrap(), 4);
        assert_eq!(
            interp.read_fd(fd, 4).unwrap(),
            IoPoll::Ready(b"ef".to_vec())
        );
        close(&mut interp, fd).unwrap();
        assert!(matches!(
            file_state(&interp, fd),
            Err(SyscallError::Descriptor(DescriptorError::InvalidFd))
        ));
    }

    #[test]
    fn unlink_keeps_open_file_live_until_last_close() {
        let mut interp = Interp::new();
        interp.vfs.write("/", "/work/temp", b"abc", 0o600).unwrap();
        let fd = open_file(
            &mut interp,
            "/work",
            "temp",
            OpenFile {
                readable: true,
                writable: true,
                create: false,
                exclusive: false,
                truncate: false,
                append: false,
            },
        )
        .unwrap();
        unlink(&mut interp, "/work", "temp").unwrap();
        assert!(!interp.vfs.lexists("/", "/work/temp"));
        assert_eq!(
            interp.read_fd(fd, 3).unwrap(),
            IoPoll::Ready(b"abc".to_vec())
        );
        seek(&mut interp, fd, 0, 0).unwrap();
        assert_eq!(interp.write_fd(fd, b"xy").unwrap(), IoPoll::Ready(2));
        seek(&mut interp, fd, 0, 0).unwrap();
        assert_eq!(
            interp.read_fd(fd, 3).unwrap(),
            IoPoll::Ready(b"xyc".to_vec())
        );
        let before_close = interp.vfs.disk_used();
        close(&mut interp, fd).unwrap();
        assert!(interp.vfs.disk_used() < before_close);
        assert!(!interp.vfs.lexists("/", "/work/temp"));
    }

    #[test]
    fn unlinking_open_symlink_keeps_target_and_descriptor() {
        let mut interp = Interp::new();
        interp.vfs.write("/", "/work/target", b"ok", 0o600).unwrap();
        interp.vfs.symlink("/", "target", "/work/link").unwrap();
        let fd = open_file(
            &mut interp,
            "/work",
            "link",
            OpenFile {
                readable: true,
                writable: false,
                create: false,
                exclusive: false,
                truncate: false,
                append: false,
            },
        )
        .unwrap();
        unlink(&mut interp, "/work", "link").unwrap();
        assert!(!interp.vfs.lexists("/", "/work/link"));
        assert!(interp.vfs.lexists("/", "/work/target"));
        assert_eq!(
            interp.read_fd(fd, 2).unwrap(),
            IoPoll::Ready(b"ok".to_vec())
        );
    }
}
