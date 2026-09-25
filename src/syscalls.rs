//! Process-owned filesystem operations shared by native execution and guest ABI adapters.
//!
//! Paths resolve only in the virtual filesystem. Open files live in the process descriptor
//! table, so a guest cannot keep a second, unaccounted set of file handles or cursors.

use crate::descriptors::{DescriptorError, Fd, FileState, IoPoll, MAX_FDS_PER_PROCESS};
use crate::display::{DisplayError, KeyEvent};
use crate::interp::Interp;
use crate::vfs::{resolve_against, NodeKind, VfsError};

/// PID-scoped virtual kernel operations available during one execution quantum.
///
/// Native programs call this interface directly. A guest ABI adapter must translate its imports
/// to the same operations; neither path receives host capabilities or the owning `Environment`.
/// The borrowed handle cannot outlive the quantum, so blocked programs retain only owned state.
pub(crate) trait System {
    fn cwd(&self) -> &str;
    fn chdir(&mut self, path: &str) -> Result<(), SyscallError>;
    fn umask(&self) -> u16;
    fn set_umask(&mut self, mask: u16) -> Result<(), SyscallError>;
    fn limits(&self) -> crate::resources::Limits;
    fn metadata(&mut self, base: &str, path: &str, follow: bool) -> Result<FileInfo, SyscallError>;
    fn metadata_fd(&mut self, fd: Fd) -> Result<FileInfo, SyscallError>;
    fn list_dir(&mut self, base: &str, path: &str) -> Result<Vec<String>, SyscallError>;
    fn walk(&mut self, base: &str, path: &str) -> Result<Vec<String>, SyscallError>;
    fn read(&mut self, fd: Fd, maximum: usize) -> Result<IoPoll<Vec<u8>>, SyscallError>;
    fn write(&mut self, fd: Fd, bytes: &[u8]) -> Result<IoPoll<usize>, SyscallError>;
    fn open_file(&mut self, base: &str, path: &str, options: OpenFile) -> Result<Fd, SyscallError>;
    fn file_state(&self, fd: Fd) -> Result<FileState, SyscallError>;
    fn close(&mut self, fd: Fd) -> Result<(), SyscallError>;
    fn seek(&mut self, fd: Fd, delta: i64, whence: u32) -> Result<u64, SyscallError>;
    fn chmod(&mut self, base: &str, path: &str, mode: u32) -> Result<(), SyscallError>;
    fn unlink(&mut self, base: &str, path: &str) -> Result<(), SyscallError>;
    fn mkdir(&mut self, base: &str, path: &str) -> Result<(), SyscallError>;
    fn mkdir_all(&mut self, base: &str, path: &str) -> Result<(), SyscallError>;
    fn rmdir(&mut self, base: &str, path: &str) -> Result<(), SyscallError>;
    fn rename(&mut self, base: &str, from: &str, to: &str) -> Result<(), SyscallError>;
    fn symlink(&mut self, base: &str, target: &str, link: &str) -> Result<(), SyscallError>;
    fn chown(
        &mut self,
        base: &str,
        path: &str,
        uid: Option<u32>,
        gid: Option<u32>,
    ) -> Result<(), SyscallError>;
    fn touch(&mut self, base: &str, path: &str, mtime_ms: u64) -> Result<(), SyscallError>;
    fn read_link(&mut self, base: &str, path: &str) -> Result<String, SyscallError>;
    fn canonicalize(
        &mut self,
        base: &str,
        path: &str,
        strict: bool,
    ) -> Result<String, SyscallError>;
    fn wall_time_ms(&self) -> u64;
    fn display_open(&mut self, width: u32, height: u32, format: u32) -> Result<u32, DisplayError>;
    fn display_present(
        &mut self,
        handle: u32,
        pixels: &[u8],
        stride: u32,
    ) -> Result<(), DisplayError>;
    fn input_poll_key(&mut self, handle: u32) -> Result<Option<KeyEvent>, DisplayError>;
    fn display_close(&mut self, handle: u32) -> Result<(), DisplayError>;
    fn charge_cpu(&mut self, units: u64) -> bool;
    fn output_remaining(&self) -> u64;
    fn charge_output(&mut self, bytes: u64) -> bool;
    fn stop_status(&self) -> i32;
}

/// Active-PID adapter; native program bodies receive this handle, not `Interp`.
/// The current WASI adapter still holds `Interp` while translating its imports.
pub(crate) struct ActiveSystem<'a> {
    interp: &'a mut Interp,
}

impl<'a> ActiveSystem<'a> {
    pub(crate) fn new(interp: &'a mut Interp) -> Self {
        Self { interp }
    }
}

impl System for ActiveSystem<'_> {
    fn cwd(&self) -> &str {
        &self.interp.process.cwd
    }

    fn chdir(&mut self, path: &str) -> Result<(), SyscallError> {
        if path.is_empty() || path.contains('\0') {
            return Err(SyscallError::InvalidArgument);
        }
        let absolute = resolve_against(&self.interp.process.cwd, path);
        let node = self.interp.fs_metadata("/", &absolute, true)?;
        if !matches!(node.kind, NodeKind::Dir) {
            return Err(SyscallError::File(VfsError::NotADir(absolute)));
        }
        let resolved = self.interp.fs_realpath("/", &absolute, true)?;
        self.interp.process.cwd = resolved;
        Ok(())
    }

    fn umask(&self) -> u16 {
        self.interp.process.umask
    }

    fn set_umask(&mut self, mask: u16) -> Result<(), SyscallError> {
        if mask > 0o777 {
            return Err(SyscallError::InvalidArgument);
        }
        self.interp.process.umask = mask;
        Ok(())
    }

    fn limits(&self) -> crate::resources::Limits {
        self.interp.resources.limits()
    }

    fn metadata(&mut self, base: &str, path: &str, follow: bool) -> Result<FileInfo, SyscallError> {
        Ok(FileInfo::from(self.interp.fs_metadata(base, path, follow)?))
    }

    fn metadata_fd(&mut self, fd: Fd) -> Result<FileInfo, SyscallError> {
        let file = file_state(self.interp, fd)?;
        let node = match file.orphan {
            Some(id) => self.interp.vfs.orphan_metadata(id)?,
            None => self.interp.fs_metadata("/", &file.path, true)?,
        };
        Ok(FileInfo::from(node))
    }

    fn list_dir(&mut self, base: &str, path: &str) -> Result<Vec<String>, SyscallError> {
        Ok(self.interp.fs_list_dir(base, path)?)
    }

    fn walk(&mut self, base: &str, path: &str) -> Result<Vec<String>, SyscallError> {
        Ok(self.interp.fs_walk(base, path)?)
    }

    fn read(&mut self, fd: Fd, maximum: usize) -> Result<IoPoll<Vec<u8>>, SyscallError> {
        self.interp.read_fd_checked(fd, maximum)
    }

    fn write(&mut self, fd: Fd, bytes: &[u8]) -> Result<IoPoll<usize>, SyscallError> {
        self.interp.write_fd_checked(fd, bytes)
    }

    fn open_file(&mut self, base: &str, path: &str, options: OpenFile) -> Result<Fd, SyscallError> {
        open_file(self.interp, base, path, options)
    }

    fn file_state(&self, fd: Fd) -> Result<FileState, SyscallError> {
        file_state(self.interp, fd)
    }

    fn close(&mut self, fd: Fd) -> Result<(), SyscallError> {
        close(self.interp, fd)
    }

    fn seek(&mut self, fd: Fd, delta: i64, whence: u32) -> Result<u64, SyscallError> {
        seek(self.interp, fd, delta, whence)
    }

    fn chmod(&mut self, base: &str, path: &str, mode: u32) -> Result<(), SyscallError> {
        chmod(self.interp, base, path, mode)
    }

    fn unlink(&mut self, base: &str, path: &str) -> Result<(), SyscallError> {
        unlink(self.interp, base, path)
    }

    fn mkdir(&mut self, base: &str, path: &str) -> Result<(), SyscallError> {
        mkdir(self.interp, base, path)
    }

    fn mkdir_all(&mut self, base: &str, path: &str) -> Result<(), SyscallError> {
        self.interp.sync_vfs_time();
        self.interp.vfs.mkdir_all(base, path)?;
        Ok(())
    }

    fn rmdir(&mut self, base: &str, path: &str) -> Result<(), SyscallError> {
        rmdir(self.interp, base, path)
    }

    fn rename(&mut self, base: &str, from: &str, to: &str) -> Result<(), SyscallError> {
        rename(self.interp, base, from, to)
    }

    fn symlink(&mut self, base: &str, target: &str, link: &str) -> Result<(), SyscallError> {
        self.interp.sync_vfs_time();
        self.interp.vfs.symlink(base, target, link)?;
        Ok(())
    }

    fn chown(
        &mut self,
        base: &str,
        path: &str,
        uid: Option<u32>,
        gid: Option<u32>,
    ) -> Result<(), SyscallError> {
        self.interp.sync_vfs_time();
        self.interp.vfs.chown(base, path, uid, gid)?;
        Ok(())
    }

    fn touch(&mut self, base: &str, path: &str, mtime_ms: u64) -> Result<(), SyscallError> {
        self.interp.sync_vfs_time();
        self.interp.vfs.touch(base, path, mtime_ms)?;
        Ok(())
    }

    fn read_link(&mut self, base: &str, path: &str) -> Result<String, SyscallError> {
        Ok(self.interp.fs_read_link(base, path)?)
    }

    fn canonicalize(
        &mut self,
        base: &str,
        path: &str,
        strict: bool,
    ) -> Result<String, SyscallError> {
        let absolute = resolve_against(base, path);
        if strict {
            Ok(self.interp.fs_realpath("/", &absolute, true)?)
        } else {
            Ok(self.interp.vfs.realpath(&absolute, true)?)
        }
    }

    fn wall_time_ms(&self) -> u64 {
        self.interp.clock.unix_ms()
    }

    fn display_open(&mut self, width: u32, height: u32, format: u32) -> Result<u32, DisplayError> {
        display_open(self.interp, width, height, format)
    }

    fn display_present(
        &mut self,
        handle: u32,
        pixels: &[u8],
        stride: u32,
    ) -> Result<(), DisplayError> {
        display_present(self.interp, handle, pixels, stride)
    }

    fn input_poll_key(&mut self, handle: u32) -> Result<Option<KeyEvent>, DisplayError> {
        input_poll_key(self.interp, handle)
    }

    fn display_close(&mut self, handle: u32) -> Result<(), DisplayError> {
        display_close(self.interp, handle)
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
fn display_open(
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
fn display_present(
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
fn input_poll_key(interp: &mut Interp, handle: u32) -> Result<Option<KeyEvent>, DisplayError> {
    interp
        .display
        .poll_key(&mut interp.resources, interp.process.pid, handle)
}

/// Relinquish the display while retaining its last frame for inspection.
fn display_close(interp: &mut Interp, handle: u32) -> Result<(), DisplayError> {
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

/// Data visible from `stat` without exposing file bytes or native program identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileInfo {
    pub kind: FileKind,
    pub mode: u32,
    pub size: u64,
    pub link_target: Option<String>,
    pub mtime_ms: u64,
    pub uid: u32,
    pub gid: u32,
    pub native_executable: bool,
}

/// Kinds observable through the virtual filesystem interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileKind {
    File,
    Directory,
    Symlink,
}

impl From<crate::vfs::Node> for FileInfo {
    fn from(node: crate::vfs::Node) -> Self {
        let (kind, size, link_target, native_executable) = match node.kind {
            NodeKind::File(data) => (FileKind::File, data.len() as u64, None, false),
            NodeKind::Dir => (FileKind::Directory, 0, None, false),
            NodeKind::Symlink(target) => {
                let size = target.len() as u64;
                (FileKind::Symlink, size, Some(target), false)
            }
            NodeKind::NativeExecutable(_) => (FileKind::File, 0, None, true),
        };
        Self {
            kind,
            mode: node.mode,
            size,
            link_target,
            mtime_ms: node.mtime,
            uid: node.uid,
            gid: node.gid,
            native_executable,
        }
    }
}

/// Typed failure at the simulated process/kernel boundary.
#[derive(Debug)]
pub(crate) enum SyscallError {
    File(VfsError),
    Descriptor(DescriptorError),
    InvalidArgument,
    IsDirectory,
    Permission,
    ResourceExhausted,
}

impl std::fmt::Display for SyscallError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::File(error) => error.fmt(formatter),
            Self::Descriptor(error) => write!(formatter, "{error:?}"),
            Self::InvalidArgument => write!(formatter, "invalid argument"),
            Self::IsDirectory => write!(formatter, "is a directory"),
            Self::Permission => write!(formatter, "permission denied"),
            Self::ResourceExhausted => write!(formatter, "resource exhausted"),
        }
    }
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
fn open_file(
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
    if absolute == "/dev/null" || crate::pseudo_fs::device_kind("/", &absolute).is_some() {
        if options.create && options.exclusive {
            return Err(SyscallError::File(VfsError::Exists(absolute)));
        }
        let description = match crate::pseudo_fs::device_kind("/", &absolute) {
            Some(kind) => {
                interp
                    .descriptors
                    .open_device(kind, options.readable, options.writable)?
            }
            None => interp.descriptors.open_null()?,
        };
        interp.install_new_description(fd, description)?;
        return Ok(fd);
    }
    if let Some(contents) = crate::pseudo_fs::read(interp, "/", &absolute) {
        if !options.readable || options.writable || options.create || options.truncate {
            return Err(SyscallError::Permission);
        }
        let description = interp.descriptors.open_input(contents?)?;
        interp.install_new_description(fd, description)?;
        return Ok(fd);
    }
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
fn chmod(interp: &mut Interp, cwd: &str, path: &str, mode: u32) -> Result<(), SyscallError> {
    if path.is_empty() || path.contains('\0') || mode & !0o7777 != 0 {
        return Err(SyscallError::InvalidArgument);
    }
    interp.sync_vfs_time();
    interp.vfs.chmod(cwd, path, mode)?;
    Ok(())
}

/// Return regular-file state for an active process descriptor.
fn file_state(interp: &Interp, fd: Fd) -> Result<FileState, SyscallError> {
    let description = interp.process.fds.get(fd)?;
    interp
        .descriptors
        .file_state(description)?
        .ok_or(SyscallError::Descriptor(DescriptorError::WrongAccess))
}

/// Close a process-owned descriptor and update process metadata.
fn close(interp: &mut Interp, fd: Fd) -> Result<(), SyscallError> {
    interp.process.fds.close(fd, &mut interp.descriptors)?;
    interp
        .vfs
        .retain_orphans(&interp.descriptors.live_orphans());
    interp.refresh_descriptor_snapshot(interp.process.pid);
    Ok(())
}

/// Remove one regular file through the virtual filesystem, without exposing host paths.
fn unlink(interp: &mut Interp, cwd: &str, path: &str) -> Result<(), SyscallError> {
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
fn mkdir(interp: &mut Interp, cwd: &str, path: &str) -> Result<(), SyscallError> {
    interp.sync_vfs_time();
    interp.vfs.mkdir(cwd, path).map_err(Into::into)
}

/// Remove only an empty virtual directory.
fn rmdir(interp: &mut Interp, cwd: &str, path: &str) -> Result<(), SyscallError> {
    interp.sync_vfs_time();
    interp.vfs.rmdir(cwd, path).map_err(Into::into)
}

/// Move a virtual path within the same modeled filesystem.
fn rename(interp: &mut Interp, cwd: &str, from: &str, to: &str) -> Result<(), SyscallError> {
    interp.sync_vfs_time();
    interp.vfs.rename(cwd, from, to).map_err(Into::into)
}

/// Seek one regular-file description. The returned cursor is shared with duplicated fds.
fn seek(interp: &mut Interp, fd: Fd, delta: i64, whence: u32) -> Result<u64, SyscallError> {
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

    #[test]
    fn active_system_changes_only_the_current_process_directory_and_mask() {
        let mut interp = Interp::new();
        let mut system = ActiveSystem::new(&mut interp);
        assert_eq!(system.cwd(), "/");
        assert!(system.chdir("/work").is_ok());
        assert_eq!(system.cwd(), "/work");
        assert!(system.chdir("/proc").is_ok());
        assert_eq!(system.cwd(), "/proc");
        assert!(matches!(
            system.chdir("/missing"),
            Err(SyscallError::File(VfsError::NotFound(_)))
        ));
        assert_eq!(system.cwd(), "/proc");
        assert!(matches!(
            system.set_umask(0o1000),
            Err(SyscallError::InvalidArgument)
        ));
        assert_eq!(system.umask(), 0o022);
        system.set_umask(0o077).unwrap();
        assert_eq!(system.umask(), 0o077);
    }

    #[test]
    fn active_system_reads_virtual_descriptors_without_host_access() {
        let mut interp = Interp::new();
        interp
            .vfs
            .write("/", "/work/data", b"sample", 0o644)
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
        let mut system = ActiveSystem::new(&mut interp);
        assert_eq!(system.read(fd, 3).unwrap(), IoPoll::Ready(b"sam".to_vec()));
        assert_eq!(system.read(fd, 3).unwrap(), IoPoll::Ready(b"ple".to_vec()));
        assert_eq!(system.read(fd, 3).unwrap(), IoPoll::Ready(Vec::new()));
    }

    #[test]
    fn metadata_and_directory_listing_use_the_active_virtual_process() {
        let mut interp = Interp::new();
        interp.vfs.mkdir("/", "/work/tree").unwrap();
        interp
            .vfs
            .write("/", "/work/tree/data", b"sample", 0o640)
            .unwrap();
        interp.vfs.symlink("/", "data", "/work/tree/link").unwrap();
        let mut system = ActiveSystem::new(&mut interp);
        system.chdir("/work/tree").unwrap();
        let cwd = system.cwd().to_string();
        assert_eq!(system.list_dir(&cwd, ".").unwrap(), ["data", "link"]);
        assert_eq!(system.walk(&cwd, ".").unwrap().len(), 3);
        let file = system.metadata(&cwd, "data", true).unwrap();
        assert_eq!(file.kind, FileKind::File);
        assert_eq!(file.mode, 0o640);
        assert_eq!(file.size, 6);
        let link = system.metadata(&cwd, "link", false).unwrap();
        assert_eq!(link.kind, FileKind::Symlink);
        assert_eq!(link.link_target.as_deref(), Some("data"));
        assert_eq!(system.metadata(&cwd, "link", true).unwrap(), file);
        assert!(matches!(
            system.metadata(&cwd, "missing", false),
            Err(SyscallError::File(VfsError::NotFound(_)))
        ));
    }
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
            ActiveSystem::new(&mut interp).metadata_fd(fd).unwrap().size,
            3
        );
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
