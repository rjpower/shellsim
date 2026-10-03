"""Process, environment and filesystem helpers over shellsim's modeled environment and VFS.

Everything here reaches the host only through the ``_os`` and ``_shellsim_vfs`` capability
modules. Process creation (``fork``, ``exec*``, ``spawn*``) is not modeled and raises ``OSError``
with ``ENOSYS`` so a program sees the frontier instead of a silent no-op.
"""

from _os import chdir, environ, getenv, getcwd, getpid, getppid, kill
import _shellsim_vfs
import errno as _errno
import posixpath as path
import stat as _stat
import sys as _sys
from posixpath import curdir, pardir, sep, pathsep, defpath, extsep, altsep, devnull

name = "posix"
linesep = "\n"
error = OSError

F_OK = 0
X_OK = 1
W_OK = 2
R_OK = 4

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2
SEEK_DATA = 3
SEEK_HOLE = 4

O_RDONLY = 0
O_WRONLY = 1
O_RDWR = 2
O_ACCMODE = 3
O_CREAT = 0o100
O_EXCL = 0o200
O_NOCTTY = 0o400
O_TRUNC = 0o1000
O_APPEND = 0o2000
O_NONBLOCK = 0o4000
O_NDELAY = O_NONBLOCK
O_DSYNC = 0o10000
O_ASYNC = 0o20000
O_DIRECT = 0o40000
O_LARGEFILE = 0o100000
O_DIRECTORY = 0o200000
O_NOFOLLOW = 0o400000
O_NOATIME = 0o1000000
O_CLOEXEC = 0o2000000
O_SYNC = 0o4010000
O_RSYNC = O_SYNC
O_FSYNC = O_SYNC
O_PATH = 0o10000000
O_TMPFILE = 0o20000000 | O_DIRECTORY

EX_OK = 0
EX_USAGE = 64
EX_DATAERR = 65
EX_NOINPUT = 66
EX_NOUSER = 67
EX_NOHOST = 68
EX_UNAVAILABLE = 69
EX_SOFTWARE = 70
EX_OSERR = 71
EX_OSFILE = 72
EX_CANTCREAT = 73
EX_IOERR = 74
EX_TEMPFAIL = 75
EX_PROTOCOL = 76
EX_NOPERM = 77
EX_CONFIG = 78

P_WAIT = 0
P_NOWAIT = 1
P_NOWAITO = 1
P_PID = 1
P_PGID = 2
P_ALL = 0
P_PIDFD = 3
WNOHANG = 1
WUNTRACED = 2
WCONTINUED = 8
WEXITED = 4
WSTOPPED = 2
WNOWAIT = 0x01000000
PRIO_PROCESS = 0
PRIO_PGRP = 1
PRIO_USER = 2
SCHED_OTHER = 0
SCHED_FIFO = 1
SCHED_RR = 2
SCHED_BATCH = 3
SCHED_IDLE = 5
SCHED_RESET_ON_FORK = 0x40000000
RTLD_LAZY = 1
RTLD_NOW = 2
RTLD_GLOBAL = 256
RTLD_LOCAL = 0
RTLD_NODELETE = 4096
RTLD_NOLOAD = 4
RTLD_DEEPBIND = 8
GRND_NONBLOCK = 1
GRND_RANDOM = 2
TMP_MAX = 238328
NGROUPS_MAX = 65536

supports_bytes_environ = False
supports_dir_fd = set()
supports_effective_ids = set()
supports_fd = set()
supports_follow_symlinks = set()

_umask = 0o022
_uid = 1000
_gid = 1000


class PathLike:
    """Abstract base for objects representing a filesystem path via ``__fspath__``."""

    def __fspath__(self):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "__fspath__")


def fspath(value):
    if isinstance(value, (str, bytes)):
        return value
    if not hasattr(value, "__fspath__"):
        raise TypeError(
            "expected str, bytes or os.PathLike object, not " + type(value).__name__
        )
    result = value.__fspath__()
    if not isinstance(result, (str, bytes)):
        raise TypeError("expected __fspath__() to return str or bytes")
    return result


def _text_path(value):
    value = fspath(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def fsencode(filename):
    filename = fspath(filename)
    if isinstance(filename, str):
        return filename.encode("utf-8")
    return filename


def fsdecode(filename):
    filename = fspath(filename)
    if isinstance(filename, bytes):
        return filename.decode("utf-8")
    return filename


def _unsupported(function, code=None):
    code = _errno.ENOSYS if code is None else code
    return OSError(code, function + " is not supported by shellsim")


# ---- environment ----


def getenvb(key, default=None):
    value = getenv(key.decode("utf-8"))
    return default if value is None else value.encode("utf-8")


def putenv(key, value):
    raise _unsupported("os.putenv", _errno.ENOTSUP)


def unsetenv(key):
    raise _unsupported("os.unsetenv", _errno.ENOTSUP)


def get_exec_path(env=None):
    if env is not None:
        value = env.get("PATH")
    else:
        value = getenv("PATH")
    if value is None:
        value = defpath
    return value.split(pathsep)


def getcwdb():
    return getcwd().encode("utf-8")


# ---- process identity ----


def getuid():
    return _uid


def geteuid():
    return _uid


def getgid():
    return _gid


def getegid():
    return _gid


def getgroups():
    return [_gid]


def getgrouplist(user, group):
    return [group]


def getresuid():
    return (_uid, _uid, _uid)


def getresgid():
    return (_gid, _gid, _gid)


def getlogin():
    user = getenv("USER")
    if user is None:
        raise OSError(_errno.ENOTTY, "Inappropriate ioctl for device")
    return user


def getpgid(pid):
    return pid


def getpgrp():
    return getpid()


def getsid(pid):
    return pid


def setsid():
    raise _unsupported("os.setsid", _errno.EPERM)


def _set_id(*ids):
    raise _unsupported("changing process ids", _errno.EPERM)


setuid = seteuid = setgid = setegid = setreuid = setregid = setresuid = setresgid = _set_id
setgroups = initgroups = setpgid = setpgrp = _set_id


def umask(mask):
    global _umask
    previous = _umask
    _umask = mask & 0o777
    return previous


def nice(increment):
    return 0


def getpriority(which, who):
    return 0


def setpriority(which, who, priority):
    return None


def cpu_count():
    return 1


def process_cpu_count():
    return 1


def sched_getaffinity(pid):
    return set(range(cpu_count()))


def sched_setaffinity(pid, mask):
    return None


def sched_yield():
    return None


def sched_get_priority_min(policy):
    return 0


def sched_get_priority_max(policy):
    return 0 if policy in (SCHED_OTHER, SCHED_BATCH, SCHED_IDLE) else 99


def getloadavg():
    return (0.0, 0.0, 0.0)


class uname_result(tuple):
    _fields = ("sysname", "nodename", "release", "version", "machine")

    def __new__(cls, sysname, nodename, release, version, machine):
        return tuple.__new__(cls, (sysname, nodename, release, version, machine))

    sysname = property(lambda self: self[0])
    nodename = property(lambda self: self[1])
    release = property(lambda self: self[2])
    version = property(lambda self: self[3])
    machine = property(lambda self: self[4])

    def __repr__(self):
        return "posix.uname_result(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(self._fields, self)
        ) + ")"


def uname():
    return uname_result("Linux", "shellsim", "6.1.0-shellsim", "#1 SMP shellsim", "x86_64")


class times_result(tuple):
    _fields = ("user", "system", "children_user", "children_system", "elapsed")

    def __new__(cls, user, system, children_user, children_system, elapsed):
        return tuple.__new__(cls, (user, system, children_user, children_system, elapsed))

    user = property(lambda self: self[0])
    system = property(lambda self: self[1])
    children_user = property(lambda self: self[2])
    children_system = property(lambda self: self[3])
    elapsed = property(lambda self: self[4])

    def __repr__(self):
        return "posix.times_result(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(self._fields, self)
        ) + ")"


def times():
    import time

    return times_result(time.process_time(), 0.0, 0.0, 0.0, time.monotonic())


class terminal_size(tuple):
    def __new__(cls, columns_lines):
        return tuple.__new__(cls, tuple(columns_lines))

    columns = property(lambda self: self[0])
    lines = property(lambda self: self[1])

    def __repr__(self):
        return "os.terminal_size(columns=%d, lines=%d)" % (self[0], self[1])


def get_terminal_size(fd=1):
    columns = getenv("COLUMNS")
    lines = getenv("LINES")
    return terminal_size(
        (int(columns) if columns else 80, int(lines) if lines else 24)
    )


def isatty(fd):
    return False


def ttyname(fd):
    raise OSError(_errno.ENOTTY, "Inappropriate ioctl for device")


def ctermid():
    return "/dev/tty"


def strerror(code):
    messages = {
        _errno.EPERM: "Operation not permitted",
        _errno.ENOENT: "No such file or directory",
        _errno.ESRCH: "No such process",
        _errno.EINTR: "Interrupted system call",
        _errno.EIO: "Input/output error",
        _errno.EBADF: "Bad file descriptor",
        _errno.ECHILD: "No child processes",
        _errno.EAGAIN: "Resource temporarily unavailable",
        _errno.ENOMEM: "Cannot allocate memory",
        _errno.EACCES: "Permission denied",
        _errno.EEXIST: "File exists",
        _errno.ENOTDIR: "Not a directory",
        _errno.EISDIR: "Is a directory",
        _errno.EINVAL: "Invalid argument",
        _errno.ENOSPC: "No space left on device",
        _errno.EPIPE: "Broken pipe",
        _errno.ERANGE: "Numerical result out of range",
        _errno.ENAMETOOLONG: "File name too long",
        _errno.ENOSYS: "Function not implemented",
        _errno.ENOTEMPTY: "Directory not empty",
        _errno.ELOOP: "Too many levels of symbolic links",
        _errno.ENOTSUP: "Operation not supported",
        _errno.ETIMEDOUT: "Connection timed out",
        _errno.ECONNREFUSED: "Connection refused",
        _errno.ENOTTY: "Inappropriate ioctl for device",
    }
    return messages.get(code, "Unknown error " + str(code))


# ---- processes ----


def fork():
    raise _unsupported("os.fork")


def forkpty():
    raise _unsupported("os.forkpty")


def _exec(*args, **kwargs):
    raise _unsupported("os.exec*")


execl = execle = execlp = execlpe = execv = execve = execvp = execvpe = _exec


def _spawn(*args, **kwargs):
    raise _unsupported("os.spawn*")


spawnl = spawnle = spawnlp = spawnlpe = spawnv = spawnve = spawnvp = spawnvpe = _spawn
posix_spawn = posix_spawnp = _spawn


def wait():
    raise ChildProcessError(_errno.ECHILD, "No child processes")


def waitpid(pid, options):
    raise ChildProcessError(_errno.ECHILD, "No child processes")


def wait3(options):
    raise ChildProcessError(_errno.ECHILD, "No child processes")


def wait4(pid, options):
    raise ChildProcessError(_errno.ECHILD, "No child processes")


def waitid(idtype, id, options):
    raise ChildProcessError(_errno.ECHILD, "No child processes")


def waitstatus_to_exitcode(status):
    if WIFEXITED(status):
        return WEXITSTATUS(status)
    if WIFSIGNALED(status):
        return -WTERMSIG(status)
    raise ValueError("invalid wait status: " + repr(status))


def WCOREDUMP(status):
    return bool(status & 0x80)


def WIFCONTINUED(status):
    return status == 0xFFFF


def WIFSTOPPED(status):
    return (status & 0xFF) == 0x7F


def WIFSIGNALED(status):
    return (status & 0x7F) != 0 and (status & 0x7F) != 0x7F


def WIFEXITED(status):
    return (status & 0x7F) == 0


def WEXITSTATUS(status):
    return (status >> 8) & 0xFF


def WSTOPSIG(status):
    return (status >> 8) & 0xFF


def WTERMSIG(status):
    return status & 0x7F


def killpg(pgid, signal):
    kill(pgid, signal)


def abort():
    kill(getpid(), 6)


def system(command):
    import subprocess

    return subprocess.call(command, shell=True) << 8


class _PopenFile:
    def __init__(self, process, stream):
        self._process = process
        self._stream = stream

    def read(self, size=-1):
        return self._stream.read(size)

    def readline(self, size=-1):
        return self._stream.readline(size)

    def readlines(self):
        return self._stream.readlines()

    def write(self, data):
        return self._stream.write(data)

    def flush(self):
        self._stream.flush()

    def __iter__(self):
        return iter(self._stream)

    def close(self):
        self._stream.close()
        status = self._process.wait()
        return None if status == 0 else status << 8

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def popen(cmd, mode="r", buffering=-1):
    import subprocess

    if not isinstance(cmd, str):
        raise TypeError("invalid cmd type (%s, expected string)" % type(cmd).__name__)
    if mode not in ("r", "w"):
        raise ValueError("invalid mode " + repr(mode))
    if mode == "r":
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)
        return _PopenFile(process, process.stdout)
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, text=True)
    return _PopenFile(process, process.stdin)


def register_at_fork(*, before=None, after_in_parent=None, after_in_child=None):
    return None


_entropy = None


def urandom(size):
    """Modeled entropy: a private generator seeded once from the virtual clock and pid, so
    it never consumes the ``random`` module's default stream."""
    global _entropy
    if size < 0:
        raise ValueError("negative argument not allowed")
    if _entropy is None:
        import random

        _entropy = random.Random()
    return _entropy.randbytes(size)


def getrandom(size, flags=0):
    return urandom(size)


# ---- filesystem ----


class stat_result(tuple):
    _fields = (
        "st_mode",
        "st_ino",
        "st_dev",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_atime",
        "st_mtime",
        "st_ctime",
    )
    n_fields = 10
    n_sequence_fields = 10
    n_unnamed_fields = 0

    def __new__(cls, values):
        return tuple.__new__(cls, tuple(values))

    st_mode = property(lambda self: self[0])
    st_ino = property(lambda self: self[1])
    st_dev = property(lambda self: self[2])
    st_nlink = property(lambda self: self[3])
    st_uid = property(lambda self: self[4])
    st_gid = property(lambda self: self[5])
    st_size = property(lambda self: self[6])
    st_atime = property(lambda self: self[7])
    st_mtime = property(lambda self: self[8])
    st_ctime = property(lambda self: self[9])
    st_atime_ns = property(lambda self: int(self[7] * 1000) * 1000000)
    st_mtime_ns = property(lambda self: int(self[8] * 1000) * 1000000)
    st_ctime_ns = property(lambda self: int(self[9] * 1000) * 1000000)
    st_blksize = property(lambda self: 4096)
    st_blocks = property(lambda self: (self[6] + 511) // 512)
    st_rdev = property(lambda self: 0)

    def __repr__(self):
        return "os.stat_result(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(self._fields, self)
        ) + ")"


_KIND_BITS = (_stat.S_IFREG, _stat.S_IFDIR, _stat.S_IFLNK, _stat.S_IFREG)


def _stat_result(path_text, follow):
    mode, size, mtime_ms, kind = _shellsim_vfs.stat(path_text, follow)
    seconds = mtime_ms / 1000.0
    full_mode = _KIND_BITS[kind] | (mode & 0o7777)
    inode = abs(hash(path.realpath(path_text) if follow else path_text)) % (1 << 32)
    return stat_result(
        (full_mode, inode, 1, 1, _uid, _gid, size, seconds, seconds, seconds)
    )


def stat(path, *, dir_fd=None, follow_symlinks=True):
    return _stat_result(_text_path(path), follow_symlinks)


def lstat(path, *, dir_fd=None):
    return _stat_result(_text_path(path), False)


def fstat(fd):
    raise _unsupported("os.fstat", _errno.EBADF)


def access(path, mode, *, dir_fd=None, effective_ids=False, follow_symlinks=True):
    path = _text_path(path)
    if not _shellsim_vfs.exists(path):
        return False
    if mode == F_OK:
        return True
    permissions = _shellsim_vfs.stat(path, follow_symlinks)[0]
    if mode & R_OK and permissions & 0o444 == 0:
        return False
    if mode & W_OK and permissions & 0o222 == 0:
        return False
    if mode & X_OK and permissions & 0o111 == 0:
        return False
    return True


def mkdir(path, mode=0o777, *, dir_fd=None):
    _shellsim_vfs.mkdir(_text_path(path), False, False)


def makedirs(name, mode=0o777, exist_ok=False):
    _shellsim_vfs.mkdir(_text_path(name), True, exist_ok)


def rmdir(path, *, dir_fd=None):
    _shellsim_vfs.rmdir(_text_path(path))


def removedirs(name):
    name = _text_path(name)
    rmdir(name)
    head, tail = path.split(name)
    if not tail:
        head, tail = path.split(head)
    while head and tail:
        try:
            rmdir(head)
        except OSError:
            break
        head, tail = path.split(head)


def listdir(path="."):
    return _shellsim_vfs.list_dir(_text_path(path))


class DirEntry:
    def __init__(self, directory, name):
        self.name = name
        self.path = path.join(directory, name) if directory != "." else name

    def __fspath__(self):
        return self.path

    def __repr__(self):
        return "<DirEntry " + repr(self.name) + ">"

    def inode(self):
        return self.stat(follow_symlinks=False).st_ino

    def is_dir(self, *, follow_symlinks=True):
        if not follow_symlinks and self.is_symlink():
            return False
        return _shellsim_vfs.is_dir(self.path)

    def is_file(self, *, follow_symlinks=True):
        if not follow_symlinks and self.is_symlink():
            return False
        return _shellsim_vfs.is_file(self.path)

    def is_symlink(self):
        return _shellsim_vfs.is_symlink(self.path)

    def is_junction(self):
        return False

    def stat(self, *, follow_symlinks=True):
        return _stat_result(self.path, follow_symlinks)


class _ScandirIterator:
    def __init__(self, directory):
        self._directory = directory
        self._names = iter(_shellsim_vfs.list_dir(directory))

    def __iter__(self):
        return self

    def __next__(self):
        return DirEntry(self._directory, next(self._names))

    def close(self):
        self._names = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def scandir(path="."):
    return _ScandirIterator(_text_path(path))


def walk(top, topdown=True, onerror=None, followlinks=False):
    top = _text_path(top)
    try:
        names = listdir(top)
    except OSError as error:
        if onerror is not None:
            onerror(error)
        return
    directories = []
    files = []
    for name in names:
        child = path.join(top, name)
        if path.isdir(child):
            directories.append(name)
        else:
            files.append(name)
    if topdown:
        yield (top, directories, files)
    for name in directories:
        child = path.join(top, name)
        if followlinks or not path.islink(child):
            for item in walk(child, topdown, onerror, followlinks):
                yield item
    if not topdown:
        yield (top, directories, files)


def fwalk(top=".", topdown=True, onerror=None, *, follow_symlinks=False, dir_fd=None):
    for root, directories, files in walk(top, topdown, onerror, follow_symlinks):
        yield (root, directories, files, -1)


def remove(path, *, dir_fd=None):
    _shellsim_vfs.remove_file(_text_path(path))


unlink = remove


def rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
    _shellsim_vfs.rename(_text_path(src), _text_path(dst))


replace = rename


def renames(old, new):
    new = _text_path(new)
    head, tail = path.split(new)
    if head and tail and not path.exists(head):
        makedirs(head)
    rename(old, new)
    head, tail = path.split(_text_path(old))
    if head and tail:
        try:
            removedirs(head)
        except OSError:
            pass


def chmod(path, mode, *, dir_fd=None, follow_symlinks=True):
    _shellsim_vfs.chmod(_text_path(path), mode)


def lchmod(path, mode):
    _shellsim_vfs.chmod(_text_path(path), mode)


def fchmod(fd, mode):
    raise _unsupported("os.fchmod", _errno.EBADF)


def chown(path, uid, gid, *, dir_fd=None, follow_symlinks=True):
    _text_path(path)
    if uid not in (-1, _uid) or gid not in (-1, _gid):
        raise PermissionError(_errno.EPERM, "Operation not permitted")


lchown = chown


def fchown(fd, uid, gid):
    raise _unsupported("os.fchown", _errno.EBADF)


def chroot(path):
    raise PermissionError(_errno.EPERM, "Operation not permitted")


def symlink(src, dst, target_is_directory=False, *, dir_fd=None):
    _shellsim_vfs.symlink(_text_path(src), _text_path(dst))


def readlink(path, *, dir_fd=None):
    return _shellsim_vfs.readlink(_text_path(path))


def link(src, dst, *, src_dir_fd=None, dst_dir_fd=None, follow_symlinks=True):
    raise _unsupported("os.link (hard links)", _errno.EPERM)


def utime(path, times=None, *, ns=None, dir_fd=None, follow_symlinks=True):
    _shellsim_vfs.utime(_text_path(path))


def truncate(path, length):
    path = _text_path(path)
    data = _shellsim_vfs.read_bytes(path)
    if length < 0:
        raise OSError(_errno.EINVAL, "Invalid argument")
    if len(data) > length:
        data = data[:length]
    else:
        data = data + bytes(length - len(data))
    _shellsim_vfs.write_bytes(path, data)


def ftruncate(fd, length):
    raise _unsupported("os.ftruncate", _errno.EBADF)


def sync():
    return None


def fsync(fd):
    return None


def fdatasync(fd):
    return None


def mkfifo(path, mode=0o666, *, dir_fd=None):
    raise _unsupported("os.mkfifo")


def mknod(path, mode=0o600, device=0, *, dir_fd=None):
    raise _unsupported("os.mknod")


def major(device):
    return (device >> 8) & 0xFFF


def minor(device):
    return (device & 0xFF) | ((device >> 12) & 0xFFF00)


def makedev(major, minor):
    return ((major & 0xFFF) << 8) | (minor & 0xFF) | ((minor & 0xFFF00) << 12)


class statvfs_result(tuple):
    _fields = (
        "f_bsize", "f_frsize", "f_blocks", "f_bfree", "f_bavail", "f_files", "f_ffree", "f_favail",
        "f_flag", "f_namemax",
    )
    n_fields = 11
    n_sequence_fields = 10
    n_unnamed_fields = 0

    def __new__(cls, values):
        return tuple.__new__(cls, tuple(values))

    f_bsize = property(lambda self: self[0])
    f_frsize = property(lambda self: self[1])
    f_blocks = property(lambda self: self[2])
    f_bfree = property(lambda self: self[3])
    f_bavail = property(lambda self: self[4])
    f_files = property(lambda self: self[5])
    f_ffree = property(lambda self: self[6])
    f_favail = property(lambda self: self[7])
    f_flag = property(lambda self: self[8])
    f_namemax = property(lambda self: self[9])
    f_fsid = property(lambda self: 0)

    def __repr__(self):
        return "os.statvfs_result(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(self._fields, self)
        ) + ")"


_BLOCK_SIZE = 4096


def statvfs(path):
    """Filesystem statistics for the single modeled disk, in 4 KiB blocks."""
    path = _text_path(path)
    if not _shellsim_vfs.exists(path):
        raise FileNotFoundError(_errno.ENOENT, "No such file or directory", path)
    used, limit = _shellsim_vfs.disk_usage()
    blocks = limit // _BLOCK_SIZE
    free = max(limit - used, 0) // _BLOCK_SIZE
    inodes = 1 << 20
    return statvfs_result((_BLOCK_SIZE, _BLOCK_SIZE, blocks, free, free, inodes, inodes, inodes, 0, 255))


def _fd_api(*args, **kwargs):
    raise _unsupported("file descriptor functions", _errno.EBADF)


open = _fd_api
close = closerange = read = write = lseek = dup = dup2 = pipe = pipe2 = _fd_api
fdopen = pread = pwrite = readv = writev = sendfile = get_inheritable = set_inheritable = _fd_api
get_blocking = set_blocking = device_encoding = fchdir = fpathconf = _fd_api


def pathconf(path, name):
    _text_path(path)
    if name in ("PC_NAME_MAX", 3):
        return 255
    if name in ("PC_PATH_MAX", 4):
        return 4096
    raise ValueError("unrecognized configuration name")


pathconf_names = {"PC_NAME_MAX": 3, "PC_PATH_MAX": 4}
sysconf_names = {"SC_PAGE_SIZE": 30, "SC_PAGESIZE": 30, "SC_NPROCESSORS_ONLN": 84, "SC_CLK_TCK": 2}
confstr_names = {"CS_PATH": 0}


def sysconf(name):
    if name in ("SC_PAGE_SIZE", "SC_PAGESIZE", 30):
        return 4096
    if name in ("SC_NPROCESSORS_ONLN", "SC_NPROCESSORS_CONF", 84, 83):
        return cpu_count()
    if name in ("SC_CLK_TCK", 2):
        return 100
    raise ValueError("unrecognized configuration name")


def confstr(name):
    if name in ("CS_PATH", 0):
        return defpath
    raise ValueError("unrecognized configuration name")


# `os.path` must be reachable as a submodule and as this attribute.
_sys.modules["os.path"] = path
