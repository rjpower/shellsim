"""Check whole-file ownership, aliases, contention and explicit unsupported APIs."""

import errno
import fcntl
import os
import subprocess
import sys
import threading
import time


def busy(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        assert error.errno == errno.EAGAIN
    else:
        raise AssertionError("conflicting lock accepted")


with open("/work/lock", "w+") as first, open("/work/lock", "r+") as second:
    fcntl.flock(first, fcntl.LOCK_SH)
    fcntl.flock(second, fcntl.LOCK_SH)
    busy(first.fileno())
    # A failed shared-to-exclusive conversion releases the prior shared lock.
    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
    busy(first.fileno())
    fcntl.flock(second, fcntl.LOCK_UN)
    flags = fcntl.fcntl(second, fcntl.F_GETFD)
    fcntl.fcntl(second, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
    assert fcntl.fcntl(second, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    fcntl.flock(first, fcntl.LOCK_EX)
    alias = os.dup(first.fileno())
    first.close()
    busy(second.fileno())
    os.close(alias)
    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(second, fcntl.LOCK_UN)
    try:
        fcntl.ioctl(second.fileno(), 0)
    except OSError as error:
        assert error.errno == errno.ENOTSUP
    else:
        raise AssertionError("unsupported ioctl accepted")
    try:
        fcntl.flock(second, 0)
    except OSError as error:
        assert error.errno == errno.EINVAL
    else:
        raise AssertionError("invalid lock mode accepted")

try:
    fcntl.flock(255, fcntl.LOCK_EX)
except OSError as error:
    assert error.errno == errno.EBADF
else:
    raise AssertionError("invalid descriptor accepted")

with open("/work/blocking-lock", "w") as owner:
    fcntl.flock(owner, fcntl.LOCK_EX)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            'import fcntl; f=open("/work/blocking-lock", "w"); fcntl.flock(f, fcntl.LOCK_EX); print("acquired")',
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.01)
    assert child.poll() is None
    fcntl.flock(owner, fcntl.LOCK_UN)
    stdout, _ = child.communicate()
    assert child.returncode == 0 and stdout == "acquired\n"
with open("/work/rename-lock", "w") as owner:
    os.rename("/work/rename-lock", "/work/renamed-lock")
    fcntl.flock(owner, fcntl.LOCK_EX)
    with open("/work/renamed-lock", "r") as contender, open("/work/rename-lock", "w") as replacement:
        busy(contender.fileno())
        fcntl.flock(replacement, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(owner, fcntl.LOCK_UN)
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

with open("/work/thread-lock", "w") as owner:
    fcntl.flock(owner, fcntl.LOCK_EX)
    pending = os.open("/work/thread-lock", os.O_RDONLY)
    entered = threading.Event()
    errors = []

    def acquire():
        entered.set()
        try:
            fcntl.flock(pending, fcntl.LOCK_EX)
        except OSError as error:
            errors.append(error.errno)

    worker = threading.Thread(target=acquire)
    worker.start()
    entered.wait()
    time.sleep(0.01)
    # An in-flight request must not acquire a different file after descriptor reuse.
    os.close(pending)
    replacement = os.open("/work/thread-replacement", os.O_RDWR | os.O_CREAT, 0o600)
    assert replacement == pending
    fcntl.flock(owner, fcntl.LOCK_UN)
    worker.join(timeout=1)
    assert not worker.is_alive() and errors == [errno.ENOTSUP]
    fcntl.flock(replacement, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(replacement)

print("fcntl: guest locks, aliases, blocking, threading, rename and errors passed")
