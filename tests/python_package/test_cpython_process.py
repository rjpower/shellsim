"""Exercise the pinned process port with real child interpreters inside shellsim.

The opt-in bundle and adapter are built artifacts. No payload runs as a host
process; the test script is written into VFS and launched by the guest runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import shellsim


def test_cpython_subprocess_and_pipe_roundtrip_in_guest(tmp_path: Path) -> None:
    bundle = os.environ.get("SHELLSIM_PROCESS_CPYTHON_BUNDLE")
    if not bundle:
        pytest.skip("set SHELLSIM_PROCESS_CPYTHON_BUNDLE to a process enabled CPython bundle")

    runtime = shellsim.CPythonRuntime(bundle)
    environment = shellsim.Environment(cpu=20_000_000_000, memory=512 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    manifest = json.loads((Path(bundle) / "manifest.json").read_text())
    sdk = Path(manifest["source_bundle"]) / "wasi-sdk-34.0-x86_64-linux"
    signal_fixture = tmp_path / "sigpipe.wasm"
    source = Path(__file__).resolve().parents[1] / "fixtures/wasi_process/sigpipe.c"
    subprocess.run(
        [str(sdk / "bin/clang"), "--target=wasm32-wasip1", str(source), "-o", str(signal_fixture)],
        check=True,
    )
    environment.write_file("/work/sigpipe.wasm", signal_fixture.read_bytes(), mode=0o755)
    environment.write_file(
        "/work/console",
        b"#!/usr/bin/env python3\nimport os, sys\nprint(sys.argv[1], os.getpid(), os.getppid())\n",
        mode=0o755,
    )
    environment.write_file("/work/junk", b"not a Wasm executable or a shebang\n", mode=0o755)
    environment.write_file(
        "/work/fault_probe.py",
        b'import faulthandler, signal\nfaulthandler.enable()\nsignal.raise_signal(signal.SIGFPE)\nprint("unexpected return")\n',
    )
    environment.write_file(
        "/work/process_probe.py",
        b"""
import errno
import faulthandler
import os
import signal
import subprocess
import sys

read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
assert not os.get_inheritable(read_fd)
assert not os.get_inheritable(write_fd)
os.set_blocking(read_fd, False)
try:
    os.read(read_fd, 1)
except BlockingIOError:
    pass
else:
    raise AssertionError('empty nonblocking pipe did not raise')
os.close(read_fd)
os.close(write_fd)
direct = os.open('/work/process_probe.py', os.O_RDONLY | os.O_CLOEXEC)
assert not os.get_inheritable(direct)
os.close(direct)
directory = os.open('/work', os.O_RDONLY | os.O_DIRECTORY)
relative = os.open('process_probe.py', os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory)
assert not os.get_inheritable(relative)
os.close(relative)
os.close(directory)
print('pipe', flush=True)

parent_pid = os.getpid()
assert parent_pid > 0
os.environ['PARENT_ONLY'] = 'secret'
child = subprocess.run(
    [sys.executable, '-c',
     'import os,sys; print(os.getcwd(), os.getenv("LABEL"), os.getenv("PARENT_ONLY"), '
     'os.getpid(), os.getppid(), sys.stdin.read())'],
    input='hello', capture_output=True, text=True, cwd='/work',
    env={'PATH': '/usr/bin:/bin', 'LABEL': 'guest'}, check=True,
)
cwd, label, parent_only, child_pid, child_parent_pid, input_text = child.stdout.split()
assert (cwd, label, parent_only, input_text) == ('/work', 'guest', 'None', 'hello')
assert int(child_pid) != parent_pid
assert int(child_parent_pid) == parent_pid
assert child.stderr == ''
shell_child = subprocess.run('printf shell', shell=True, capture_output=True, text=True, check=True)
assert shell_child.stdout == 'shell'
assert shell_child.stderr == ''
print('run', flush=True)

payload = bytes(range(256)) * 1024
child = subprocess.Popen(
    [sys.executable, '-c',
     'import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data[::-1]); '
     'sys.stderr.write(str(len(data)))'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
stdout, stderr = child.communicate(payload, timeout=5)
assert child.returncode == 0
assert stdout == payload[::-1]
assert stderr == str(len(payload)).encode()
print('duplex', flush=True)

failed = subprocess.run([sys.executable, '-c', 'import sys; sys.exit(7)'])
assert failed.returncode == 7
numeric_signal_code = subprocess.run([sys.executable, '-c', 'import sys; sys.exit(141)'])
assert numeric_signal_code.returncode == 141
child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read(1)'],
                         stdin=subprocess.PIPE)
child.terminate()
assert child.wait(timeout=5) == -signal.SIGTERM
self_terminated = subprocess.run(
    [sys.executable, '-c', 'import os,signal; os.kill(os.getpid(), signal.SIGTERM)']
)
assert self_terminated.returncode == -signal.SIGTERM
print('status', flush=True)

console = subprocess.Popen(['/work/console', 'answer'], stdout=subprocess.PIPE, text=True)
console_stdout, _ = console.communicate(timeout=5)
argument, script_pid, script_parent_pid = console_stdout.split()
assert argument == 'answer'
assert int(script_pid) == console.pid
assert int(script_parent_pid) == parent_pid
assert console.returncode == 0
try:
    subprocess.Popen(['/work/junk'])
except OSError as error:
    assert error.errno == errno.ENOEXEC
else:
    raise AssertionError('non-executable payload launched a child')
print('exec', flush=True)

assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN
for restore, expected in ((True, -signal.SIGPIPE), (False, 3)):
    child = subprocess.Popen(['/work/sigpipe.wasm'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, restore_signals=restore)
    child.stdout.close()
    child.stdin.write(b'x')
    child.stdin.flush()
    child.stdin.close()
    assert child.wait(timeout=5) == expected
print('sigpipe', flush=True)

assert not faulthandler.is_enabled()
faulthandler.enable()
assert faulthandler.is_enabled()
faulthandler.disable()
assert not faulthandler.is_enabled()
seen = []
def callback(signum, frame):
    seen.append(signum)
signal.signal(signal.SIGFPE, callback)
signal.raise_signal(signal.SIGFPE)
assert seen == [signal.SIGFPE]
print('sync-signal', flush=True)

try:
    subprocess.run([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=0.05)
except subprocess.TimeoutExpired:
    pass
else:
    raise AssertionError('virtual timeout was ignored')
print('timeout', flush=True)

try:
    signal.signal(signal.SIGINT, lambda *_: None)
except OSError as error:
    assert error.errno == errno.ENOTSUP
else:
    raise AssertionError('unsupported signal callback was accepted')
try:
    subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
except NotImplementedError:
    pass
else:
    raise AssertionError('unsupported session option launched a child')
try:
    os.posix_spawn(sys.executable, [sys.executable] + ['x'] * 256, os.environ)
except OSError as error:
    assert error.errno == errno.E2BIG
else:
    raise AssertionError('oversized argument vector launched a child')
print('frontier', flush=True)
""",
    )
    result = environment.run("/work/.venv/bin/python /work/process_probe.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        b"pipe",
        b"run",
        b"duplex",
        b"status",
        b"exec",
        b"sigpipe",
        b"sync-signal",
        b"timeout",
        b"frontier",
    ]
    assert result.stderr == b""
    fatal = environment.run("/work/.venv/bin/python /work/fault_probe.py")
    assert fatal.returncode != 0
    assert b"Fatal Python error: Floating-point exception" in fatal.stderr
    assert b"/work/fault_probe.py" in fatal.stderr
    assert b"unexpected return" not in fatal.stdout
