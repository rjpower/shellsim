"""Check corrected libc errno across Python threads and virtual children."""

import errno
import os
import subprocess
import sys
import threading


def check_errno():
    try:
        os.read(-1, 1)
    except OSError as error:
        assert error.errno == errno.EBADF
    else:
        raise AssertionError("invalid descriptor succeeded")
    try:
        os.open("/missing-parent-tls-probe", os.O_RDONLY)
    except OSError as error:
        assert error.errno == errno.ENOENT
    else:
        raise AssertionError("missing path opened")


check_errno()
completed = []


def worker():
    check_errno()
    completed.append(True)


threads = [threading.Thread(target=worker) for _ in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
assert completed == [True, True]
check_errno()
result = subprocess.run(
    [sys.executable, "-c", "import os; print(os.getcwd()); print(os.environ['CHILD_VALUE'])"],
    cwd="/tmp",
    env={"CHILD_VALUE": "corrected-platform"},
    capture_output=True,
    check=True,
)
assert result.stdout == b"/tmp\ncorrected-platform\n"
assert result.stderr == b""
arguments = [f"arg-{index}" for index in range(1500)]
result = subprocess.run(
    [
        sys.executable,
        "-c",
        "import sys; assert sys.argv[1:] == [f'arg-{i}' for i in range(1500)]; print('large argv passed')",
        *arguments,
    ],
    capture_output=True,
    check=True,
)
assert result.stdout == b"large argv passed\n"
assert result.stderr == b""
check_errno()
print("threaded CPython: parent/worker errno and subprocess passed")
