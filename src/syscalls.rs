//! Process-owned filesystem operations shared by native execution and guest ABI adapters.
//!
//! Paths resolve only in the virtual filesystem. Open files live in the process descriptor
//! table, so a guest cannot keep a second, unaccounted set of file handles or cursors.

use crate::descriptors::{DescriptorError, Fd, FileState, MAX_FDS_PER_PROCESS};
use crate::interp::Interp;
use crate::vfs::{resolve_against, NodeKind, VfsError};

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
    let fd = (4..MAX_FDS_PER_PROCESS as Fd + 4)
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
    let cursor = if options.append {
        interp.vfs.file_len("/", &absolute)? as u64
    } else {
        0
    };
    let description = interp.descriptors.open_file(
        absolute,
        cursor,
        options.readable,
        options.writable,
        false,
    )?;
    interp.install_new_description(fd, description)?;
    Ok(fd)
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
    interp.refresh_descriptor_snapshot(interp.process.pid);
    Ok(())
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
        2 => interp.vfs.file_len("/", &file.path)? as i128,
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
}
