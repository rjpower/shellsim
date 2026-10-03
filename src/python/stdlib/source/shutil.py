"""High-level file operations confined to shellsim's modeled VFS.

Every operation goes through ``os`` and the ``_shellsim_vfs`` core, so copies, moves and
removals stay inside the simulated tree. Archive support covers the ``zip`` format through
``zipfile``; tar formats are not registered.
"""

import errno as _errno
import fnmatch
import os
import stat as _stat
import sys

__all__ = [
    "copyfileobj", "copyfile", "copymode", "copystat", "copy", "copy2", "copytree", "move",
    "rmtree", "Error", "SpecialFileError", "ExecError", "make_archive", "get_archive_formats",
    "register_archive_format", "unregister_archive_format", "get_unpack_formats",
    "register_unpack_format", "unregister_unpack_format", "unpack_archive", "ignore_patterns",
    "chown", "which", "get_terminal_size", "SameFileError", "disk_usage", "ReadError",
    "RegistryError",
]

COPY_BUFSIZE = 64 * 1024


class Error(OSError):
    pass


class SameFileError(Error):
    """Raised when source and destination are the same file."""


class SpecialFileError(OSError):
    """Raised when trying to do a kind of operation (e.g. copying) which is not supported on a
    special file (e.g. a named pipe)."""


class ExecError(OSError):
    """Raised when a command could not be executed."""


class ReadError(OSError):
    """Raised when an archive cannot be read."""


class RegistryError(Exception):
    """Raised when a registry operation with the archiving and unpacking registries fails."""


def copyfileobj(fsrc, fdst, length=0):
    if not length:
        length = COPY_BUFSIZE
    read = fsrc.read
    write = fdst.write
    while True:
        buffer = read(length)
        if not buffer:
            break
        write(buffer)


def _samefile(src, dst):
    try:
        return os.path.samefile(src, dst)
    except OSError:
        return False


def _check_special(path, name):
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return
    if _stat.S_ISFIFO(mode) or _stat.S_ISCHR(mode) or _stat.S_ISBLK(mode):
        raise SpecialFileError("`%s` is a named pipe" % (name,))


def copyfile(src, dst, *, follow_symlinks=True):
    if _samefile(src, dst):
        raise SameFileError("{!r} and {!r} are the same file".format(src, dst))
    for path, name in ((src, src), (dst, dst)):
        _check_special(path, name)
    if not follow_symlinks and os.path.islink(src):
        os.symlink(os.readlink(src), dst)
        return dst
    if os.path.isdir(src):
        raise IsADirectoryError(_errno.EISDIR, os.strerror(_errno.EISDIR), os.fspath(src))
    with open(src, "rb") as fsrc:
        with open(dst, "wb") as fdst:
            copyfileobj(fsrc, fdst)
    return dst


def copymode(src, dst, *, follow_symlinks=True):
    if not follow_symlinks and os.path.islink(src) and os.path.islink(dst):
        return
    st = os.stat(src) if follow_symlinks else os.lstat(src)
    os.chmod(dst, _stat.S_IMODE(st.st_mode))


def copystat(src, dst, *, follow_symlinks=True):
    follow = follow_symlinks or not (os.path.islink(src) and os.path.islink(dst))
    st = os.stat(src) if follow else os.lstat(src)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.chmod(dst, _stat.S_IMODE(st.st_mode))


def copy(src, dst, *, follow_symlinks=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    copyfile(src, dst, follow_symlinks=follow_symlinks)
    copymode(src, dst, follow_symlinks=follow_symlinks)
    return dst


def copy2(src, dst, *, follow_symlinks=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    copyfile(src, dst, follow_symlinks=follow_symlinks)
    copystat(src, dst, follow_symlinks=follow_symlinks)
    return dst


def ignore_patterns(*patterns):
    """A ``copytree`` ignore callable that skips names matching any glob pattern."""

    def _ignore_patterns(path, names):
        ignored = []
        for pattern in patterns:
            ignored.extend(fnmatch.filter(names, pattern))
        return set(ignored)

    return _ignore_patterns


def copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2,
             ignore_dangling_symlinks=False, dirs_exist_ok=False):
    names = os.listdir(src)
    ignored_names = ignore(os.fspath(src), names) if ignore is not None else set()
    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                target = os.readlink(srcname)
                if symlinks:
                    os.symlink(target, dstname)
                    copystat(srcname, dstname, follow_symlinks=False)
                else:
                    if not os.path.exists(srcname):
                        if ignore_dangling_symlinks:
                            continue
                        raise FileNotFoundError(_errno.ENOENT, "No such file or directory", srcname)
                    if os.path.isdir(srcname):
                        copytree(srcname, dstname, symlinks, ignore, copy_function,
                                 ignore_dangling_symlinks, dirs_exist_ok)
                    else:
                        copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore, copy_function,
                         ignore_dangling_symlinks, dirs_exist_ok)
            else:
                copy_function(srcname, dstname)
        except Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError as why:
        errors.append((src, dst, str(why)))
    if errors:
        raise Error(errors)
    return dst


def _rmtree_report(func, path, exc, onerror, onexc):
    if onexc is not None:
        onexc(func, path, exc)
    elif onerror is not None:
        onerror(func, path, (type(exc), exc, None))
    else:
        raise exc


def _rmtree_directory(path, onerror, onexc):
    try:
        names = os.listdir(path)
    except OSError as exc:
        _rmtree_report(os.listdir, path, exc, onerror, onexc)
        return
    for name in names:
        child = os.path.join(path, name)
        try:
            is_dir = os.path.isdir(child) and not os.path.islink(child)
        except OSError:
            is_dir = False
        if is_dir:
            _rmtree_directory(child, onerror, onexc)
        else:
            try:
                os.unlink(child)
            except OSError as exc:
                _rmtree_report(os.unlink, child, exc, onerror, onexc)
    try:
        os.rmdir(path)
    except OSError as exc:
        _rmtree_report(os.rmdir, path, exc, onerror, onexc)


def rmtree(path, ignore_errors=False, onerror=None, *, onexc=None, dir_fd=None):
    if dir_fd is not None:
        raise NotImplementedError("rmtree: dir_fd unavailable on this platform")
    if ignore_errors:
        def onexc(*args):
            pass
    path = os.fspath(path)
    try:
        if os.path.islink(path):
            raise OSError("Cannot call rmtree on a symbolic link")
    except OSError as exc:
        _rmtree_report(os.path.islink, path, exc, onerror, onexc)
        return
    if not os.path.isdir(path):
        try:
            raise NotADirectoryError(_errno.ENOTDIR, os.strerror(_errno.ENOTDIR), path) \
                if os.path.exists(path) else \
                FileNotFoundError(_errno.ENOENT, os.strerror(_errno.ENOENT), path)
        except OSError as exc:
            _rmtree_report(os.lstat, path, exc, onerror, onexc)
        return
    _rmtree_directory(path, onerror, onexc)


rmtree.avoids_symlink_attacks = False


def _basename(path):
    path = os.fspath(path)
    sep = os.path.sep
    return os.path.basename(path.rstrip(sep))


def move(src, dst, copy_function=copy2):
    real_dst = dst
    if os.path.isdir(dst):
        if _samefile(src, dst) and not os.path.islink(src):
            os.rename(src, dst)
            return dst
        real_dst = os.path.join(dst, _basename(src))
        if os.path.exists(real_dst):
            raise Error("Destination path '%s' already exists" % real_dst)
    try:
        os.rename(src, real_dst)
    except OSError:
        if os.path.islink(src):
            os.symlink(os.readlink(src), real_dst)
            os.unlink(src)
        elif os.path.isdir(src):
            if _destinsrc(src, dst):
                raise Error("Cannot move a directory '%s' into itself '%s'." % (src, dst))
            copytree(src, real_dst, copy_function=copy_function, symlinks=True)
            rmtree(src)
        else:
            copy_function(src, real_dst)
            os.unlink(src)
    return real_dst


def _destinsrc(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if not src.endswith(os.path.sep):
        src += os.path.sep
    if not dst.endswith(os.path.sep):
        dst += os.path.sep
    return dst.startswith(src)


class _ntuple_diskusage(tuple):
    _fields = ("total", "used", "free")

    def __new__(cls, total, used, free):
        return tuple.__new__(cls, (total, used, free))

    total = property(lambda self: self[0])
    used = property(lambda self: self[1])
    free = property(lambda self: self[2])

    def __repr__(self):
        return "usage(total=%r, used=%r, free=%r)" % tuple(self)


def disk_usage(path):
    """Usage of the modeled filesystem holding ``path``, from ``os.statvfs``."""
    st = os.statvfs(path)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return _ntuple_diskusage(total, used, free)


def chown(path, user=None, group=None, *, dir_fd=None, follow_symlinks=True):
    if user is None and group is None:
        raise ValueError("user and/or group must be set")
    uid = -1 if user is None else _lookup_id(user, os.getuid, "user")
    gid = -1 if group is None else _lookup_id(group, os.getgid, "group")
    os.chown(path, uid, gid, dir_fd=dir_fd, follow_symlinks=follow_symlinks)


def _lookup_id(value, current, kind):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value in ("root", "user", os.getenv("USER", "")):
            return current()
        raise LookupError("no such %s: %r" % (kind, value))
    raise TypeError("%s must be an int or str" % kind)


def get_terminal_size(fallback=(80, 24)):
    try:
        columns = int(os.environ["COLUMNS"])
    except (KeyError, ValueError):
        columns = 0
    try:
        lines = int(os.environ["LINES"])
    except (KeyError, ValueError):
        lines = 0
    if columns <= 0 or lines <= 0:
        try:
            size = os.get_terminal_size()
        except (AttributeError, ValueError, OSError):
            size = os.terminal_size(fallback)
        if columns <= 0:
            columns = size.columns or fallback[0]
        if lines <= 0:
            lines = size.lines or fallback[1]
    return os.terminal_size((columns, lines))


def _access_check(fn, mode):
    return os.path.exists(fn) and os.access(fn, mode) and not os.path.isdir(fn)


def which(cmd, mode=os.F_OK | os.X_OK, path=None):
    cmd = os.fspath(cmd)
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None
    if path is None:
        path = os.environ.get("PATH", None)
        if path is None:
            path = os.defpath
    if not path:
        return None
    seen = set()
    for directory in os.fspath(path).split(os.pathsep):
        normdir = os.path.normcase(directory)
        if normdir in seen:
            continue
        seen.add(normdir)
        name = os.path.join(directory, cmd)
        if _access_check(name, mode):
            return name
    return None


# ---- archives ----


def _make_zipfile(base_name, base_dir, verbose=0, dry_run=0, logger=None, owner=None,
                  group=None, root_dir=None):
    import zipfile

    zip_filename = base_name + ".zip"
    archive_dir = os.path.dirname(base_name)
    if archive_dir and not os.path.exists(archive_dir):
        if not dry_run:
            os.makedirs(archive_dir)
    if dry_run:
        return zip_filename
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        arcname = os.path.normpath(base_dir)
        if root_dir is not None:
            base_dir = os.path.join(root_dir, base_dir)
        base_dir = os.path.normpath(base_dir)
        if arcname != os.curdir:
            zf.write(base_dir, arcname)
        for dirpath, dirnames, filenames in os.walk(base_dir):
            arcdirpath = dirpath
            if root_dir is not None:
                arcdirpath = os.path.relpath(arcdirpath, root_dir)
            arcdirpath = os.path.normpath(arcdirpath)
            for name in sorted(dirnames):
                path = os.path.join(dirpath, name)
                arcname = os.path.join(arcdirpath, name)
                zf.write(path, arcname)
            for name in filenames:
                path = os.path.join(dirpath, name)
                path = os.path.normpath(path)
                if os.path.isfile(path):
                    arcname = os.path.join(arcdirpath, name)
                    zf.write(path, arcname)
    return zip_filename


_ARCHIVE_FORMATS = {
    "zip": (_make_zipfile, [], "ZIP file"),
}


def get_archive_formats():
    formats = [(name, registry[2]) for name, registry in _ARCHIVE_FORMATS.items()]
    formats.sort()
    return formats


def register_archive_format(name, function, extra_args=None, description=""):
    if extra_args is None:
        extra_args = []
    if not callable(function):
        raise TypeError("The %s object is not callable" % function)
    if not isinstance(extra_args, (tuple, list)):
        raise TypeError("extra_args needs to be a sequence")
    for element in extra_args:
        if not isinstance(element, (tuple, list)) or len(element) != 2:
            raise TypeError("extra_args elements are : (arg_name, value)")
    _ARCHIVE_FORMATS[name] = (function, extra_args, description)


def unregister_archive_format(name):
    del _ARCHIVE_FORMATS[name]


def make_archive(base_name, format, root_dir=None, base_dir=None, verbose=0, dry_run=0,
                 owner=None, group=None, logger=None):
    try:
        format_info = _ARCHIVE_FORMATS[format]
    except KeyError:
        raise ValueError("unknown archive format '%s'" % format) from None
    func = format_info[0]
    kwargs = {"dry_run": dry_run, "logger": logger, "owner": owner, "group": group}
    for arg, value in format_info[1]:
        kwargs[arg] = value
    if base_dir is None:
        base_dir = os.curdir
    base_name = os.fspath(base_name)
    if root_dir is not None:
        root_dir = os.fspath(root_dir)
        if not os.path.isabs(base_name):
            base_name = os.path.join(os.getcwd(), base_name)
        kwargs["root_dir"] = root_dir
    return func(base_name, base_dir, **kwargs)


def _ensure_directory(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def _unpack_zipfile(filename, extract_dir):
    import zipfile

    if not zipfile.is_zipfile(filename):
        raise ReadError("%s is not a zip file" % filename)
    with zipfile.ZipFile(filename) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or ".." in name.split("/"):
                continue
            targetpath = os.path.join(extract_dir, *name.split("/"))
            if not name or name.endswith("/"):
                if not os.path.isdir(targetpath):
                    os.makedirs(targetpath)
                continue
            _ensure_directory(targetpath)
            data = zf.read(info.filename)
            with open(targetpath, "wb") as target:
                target.write(data)


_UNPACK_FORMATS = {
    "zip": ([".zip"], _unpack_zipfile, [], "ZIP file"),
}


def get_unpack_formats():
    formats = [(name, info[0], info[3]) for name, info in _UNPACK_FORMATS.items()]
    formats.sort()
    return formats


def _check_unpack_options(extensions, function, extra_args):
    existing = {}
    for name, info in _UNPACK_FORMATS.items():
        for ext in info[0]:
            existing[ext] = name
    for extension in extensions:
        if extension in existing:
            raise RegistryError("%s is already registered for \"%s\"" % (extension, existing[extension]))
    if not callable(function):
        raise TypeError("The registered function must be a callable")


def register_unpack_format(name, extensions, function, extra_args=None, description=""):
    if extra_args is None:
        extra_args = []
    _check_unpack_options(extensions, function, extra_args)
    _UNPACK_FORMATS[name] = (extensions, function, extra_args, description)


def unregister_unpack_format(name):
    del _UNPACK_FORMATS[name]


def _find_unpack_format(filename):
    for name, info in _UNPACK_FORMATS.items():
        for extension in info[0]:
            if filename.endswith(extension):
                return name
    return None


def unpack_archive(filename, extract_dir=None, format=None, *, filter=None):
    filename = os.fspath(filename)
    if extract_dir is None:
        extract_dir = os.getcwd()
    extract_dir = os.fspath(extract_dir)
    if format is not None:
        try:
            format_info = _UNPACK_FORMATS[format]
        except KeyError:
            raise ValueError("Unknown unpack format '{0}'".format(format)) from None
    else:
        format = _find_unpack_format(filename)
        if format is None:
            raise ReadError("Unknown archive format '{0}'".format(filename))
        format_info = _UNPACK_FORMATS[format]
    func = format_info[1]
    kwargs = dict(format_info[2])
    func(filename, extract_dir, **kwargs)
