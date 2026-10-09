//! Atomic virtual open with per-descriptor close-on-exec intent.
//!
//! A guest import cannot yield between installing the descriptor and its flags. This
//! prevents concurrent exec from inheriting an intermediate, incorrectly flagged FD.

use super::*;

const O_CLOEXEC: u32 = 0x0008_0000;
const O_DIRECTORY: u32 = 0x0000_2000;
const O_NOFOLLOW: u32 = 0x0100_0000;
const O_RDONLY: u32 = 0x0400_0000;
const O_WRONLY: u32 = 0x1000_0000;
const O_SEARCH: u32 = 0x0800_0000;
const SUPPORTED: u32 = O_CLOEXEC
    | O_DIRECTORY
    | O_NOFOLLOW
    | O_RDONLY
    | O_WRONLY
    | O_SEARCH
    | 0x1000
    | 0x4000
    | 0x8000
    | 1
    | 4;

fn open(
    mut caller: Caller<'_, Host>,
    directory: i32,
    pointer: u32,
    flags: u32,
    mode: u32,
) -> Result<i32, Error> {
    if flags & !SUPPORTED != 0 {
        return Ok(-ERRNO_NOTSUP);
    }
    if mode & !0o7777 != 0 || flags & (O_RDONLY | O_WRONLY | O_SEARCH) == 0 {
        return Ok(-ERRNO_INVAL);
    }
    let Some(memory) = memory(&mut caller) else {
        return Ok(-ERRNO_FAULT);
    };
    {
        let mut machine = caller.data_mut().machine.get();
        if !machine.resources.charge_cpu(4097) || !machine.resources.reserve_memory(8192) {
            return Err(exhausted());
        }
    }
    let path = memory
        .c_string(&caller, pointer as usize, 4097)
        .map_err(|_| ERRNO_FAULT)
        .and_then(|bytes| match bytes {
            std::borrow::Cow::Borrowed(bytes) => std::str::from_utf8(bytes)
                .map(str::to_owned)
                .map_err(|_| ERRNO_INVAL),
            std::borrow::Cow::Owned(bytes) => String::from_utf8(bytes).map_err(|_| ERRNO_INVAL),
        });
    let result = path.and_then(|path| {
        let mut machine = caller.data_mut().machine.get();
        let base = if path.starts_with('/') || directory == -2 {
            machine.process.cwd.clone()
        } else {
            ActiveSystem::new(&mut machine)
                .directory_path(directory)
                .map_err(|error| syscall_errno(&error))?
        };
        let existing = machine.vfs.metadata(&base, &path, false);
        if flags & (0x1000 | 0x4000) == (0x1000 | 0x4000) && existing.is_ok() {
            return Err(ERRNO_EXIST);
        }
        let virtual_device = machine
            .vfs
            .realpath(&resolve_against(&base, &path), true)
            .is_ok_and(|resolved| {
                resolved == "/dev/null" || crate::pseudo_fs::device_kind("/", &resolved).is_some()
            });
        let target_missing = !virtual_device
            && matches!(
                machine.vfs.metadata(&base, &path, true),
                Err(VfsError::NotFound(_))
            );
        if flags & O_NOFOLLOW != 0
            && existing
                .as_ref()
                .is_ok_and(|node| matches!(node.kind, crate::vfs::NodeKind::Symlink(_)))
        {
            return Err(32); // WASI ELOOP: the final path component must not be a symlink.
        }
        if flags & (O_DIRECTORY | O_SEARCH) != 0 {
            match machine.vfs.metadata(&base, &path, true) {
                Ok(node) if matches!(node.kind, crate::vfs::NodeKind::Dir) => {}
                Ok(_) => return Err(ERRNO_NOTDIR),
                Err(error) => return Err(vfs_errno(&error)),
            }
        }
        let options = OpenFile {
            readable: flags & (O_RDONLY | O_SEARCH) != 0,
            writable: flags & O_WRONLY != 0,
            create: flags & 0x1000 != 0,
            exclusive: flags & 0x4000 != 0,
            truncate: flags & 0x8000 != 0,
            append: flags & 1 != 0,
        };
        let fd = ActiveSystem::new(&mut machine)
            .open_file(&base, &path, options)
            .map_err(|error| syscall_errno(&error))?;
        let setup = (|| {
            if options.create && target_missing {
                let umask = machine.process.umask;
                machine
                    .vfs
                    .chmod(&base, &path, mode & !u32::from(umask))
                    .map_err(|error| vfs_errno(&error))?;
            }
            machine
                .process
                .fds
                .set_close_on_exec(fd, flags & O_CLOEXEC != 0)
                .map_err(|_| ERRNO_BADF)?;
            let description = machine.process.fds.get(fd).map_err(|_| ERRNO_BADF)?;
            machine
                .descriptors
                .set_nonblocking(description, flags & 4 != 0)
                .map_err(|_| ERRNO_BADF)?;
            Ok(fd)
        })();
        if setup.is_err() {
            let _ = ActiveSystem::new(&mut machine).close(fd);
        }
        setup
    });
    caller
        .data_mut()
        .machine
        .get()
        .resources
        .release_memory(8192);
    match result {
        Ok(fd) => {
            caller.data_mut().open_files.insert(fd);
            Ok(fd)
        }
        Err(_) if caller.data().machine.get().resources.is_stopped() => Err(exhausted()),
        Err(errno) => Ok(-errno),
    }
}

pub(super) fn register(linker: &mut Linker<Host>) {
    linker
        .func_wrap("shellsim_posix_v1", "descriptor_open", open)
        .expect("unique descriptor open import");
}
