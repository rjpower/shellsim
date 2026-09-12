"""Cooperative subprocess compatibility over shellsim logical processes."""

import _shellsim_subprocess
from _shellsim_subprocess import CalledProcessError, TimeoutExpired


PIPE = -1
STDOUT = -2
DEVNULL = -3


class CompletedProcess:
    def __init__(self, args, returncode, stdout=None, stderr=None):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def check_returncode(self):
        if self.returncode != 0:
            raise CalledProcessError(
                "Command " + repr(self.args) + " returned non-zero exit status " + str(self.returncode)
            )

    def __repr__(self):
        result = "CompletedProcess(args=" + repr(self.args) + ", returncode=" + str(self.returncode)
        if self.stdout is not None:
            result += ", stdout=" + repr(self.stdout)
        if self.stderr is not None:
            result += ", stderr=" + repr(self.stderr)
        return result + ")"


def _text_mode(text, encoding):
    return text or encoding is not None


def _decode(value, text, encoding, errors):
    if value is None or not _text_mode(text, encoding):
        return value
    if encoding is None:
        encoding = "utf-8"
    if errors is None:
        errors = "strict"
    return value.decode(encoding, errors)


def run(args, stdin=None, input=None, stdout=None, stderr=None, capture_output=False,
        shell=False, cwd=None, env=None, timeout=None, check=False, text=False,
        encoding=None, errors=None, executable=None, universal_newlines=False):
    original_args = args
    input_supplied = input is not None
    if isinstance(args, str):
        if shell:
            args = ["sh", "-c", args]
        else:
            args = [args]
    elif shell:
        raise TypeError("shell=True requires a command string in shellsim")
    if executable is not None:
        raise ValueError("executable is not supported by shellsim subprocess")
    if capture_output:
        if stdout is not None or stderr is not None:
            raise ValueError("stdout and stderr may not be used with capture_output")
        stdout = PIPE
        stderr = PIPE
    if stdin is not None and stdin != PIPE:
        raise ValueError("stdin must be PIPE or None")
    if input is not None and stdin is not None:
        raise ValueError("stdin and input arguments may not both be used")
    text = text or universal_newlines
    if input_supplied:
        stdin = PIPE
    process = Popen(args, stdin=stdin, stdout=stdout, stderr=stderr, cwd=cwd, env=env,
                    shell=False, text=text, encoding=encoding, errors=errors)
    try:
        child_stdout, child_stderr = process.communicate(input=input, timeout=timeout)
    except TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    result = CompletedProcess(original_args, process.returncode, child_stdout, child_stderr)
    if check:
        result.check_returncode()
    return result


def call(args, stdin=None, stdout=None, stderr=None, shell=False, cwd=None, env=None,
         timeout=None):
    return run(args, stdin=stdin, stdout=stdout, stderr=stderr, shell=shell, cwd=cwd,
               env=env, timeout=timeout).returncode


def check_call(args, stdin=None, stdout=None, stderr=None, shell=False, cwd=None, env=None,
               timeout=None):
    return run(args, stdin=stdin, stdout=stdout, stderr=stderr, shell=shell, cwd=cwd,
               env=env, timeout=timeout, check=True).returncode


def check_output(args, input=None, stderr=None, shell=False, cwd=None, env=None, timeout=None,
                 text=False, encoding=None, errors=None):
    return run(args, input=input, stdout=PIPE, stderr=stderr, shell=shell, cwd=cwd, env=env,
               timeout=timeout, check=True, text=text, encoding=encoding, errors=errors).stdout


class _Pipe:
    """Small file-like facade over one parent-side modeled pipe endpoint."""

    def __init__(self, process, fd, readable):
        self._process = process
        self._fd = fd
        self._readable = readable
        self.closed = False

    def _check_open(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def read(self, size=-1):
        self._check_open()
        if not self._readable:
            raise ValueError("stream is not readable")
        value = _shellsim_subprocess.read_pipe(self._process._handle, self._fd, size)
        return _decode(value, self._process._text, self._process.encoding,
                       self._process.errors)

    def readline(self, size=-1):
        self._check_open()
        if not self._readable:
            raise ValueError("stream is not readable")
        result = "" if self._process._text else b""
        while size < 0 or len(result) < size:
            part = self.read(1)
            if not part:
                break
            result += part
            if part == ("\n" if self._process._text else b"\n"):
                break
        return result

    def write(self, value):
        self._check_open()
        if self._readable:
            raise ValueError("stream is not writable")
        if self._process._text:
            encoding = self._process.encoding if self._process.encoding is not None else "utf-8"
            errors = self._process.errors if self._process.errors is not None else "strict"
            value = value.encode(encoding, errors)
        return _shellsim_subprocess.write_pipe(self._process._handle, value)

    def flush(self):
        self._check_open()

    def close(self):
        if not self.closed:
            _shellsim_subprocess.close_pipe(self._process._handle, self._fd)
            self.closed = True

    def fileno(self):
        self._check_open()
        return self._fd

    def readable(self):
        return self._readable and not self.closed

    def writable(self):
        return not self._readable and not self.closed

    def seekable(self):
        return False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.close()


class Popen:
    def __init__(self, args, bufsize=-1, executable=None, stdin=None, stdout=None, stderr=None,
                 preexec_fn=None, close_fds=True, shell=False, cwd=None, env=None,
                 universal_newlines=None, startupinfo=None, creationflags=0, restore_signals=True,
                 start_new_session=False, pass_fds=(), user=None, group=None, extra_groups=None,
                 encoding=None, errors=None, text=None, umask=-1, pipesize=-1, process_group=None):
        self.args = args
        if isinstance(args, str):
            if shell:
                args = ["sh", "-c", args]
            else:
                args = [args]
        elif shell:
            raise TypeError("shell=True requires a command string in shellsim")
        if executable is not None:
            raise ValueError("executable is not supported by shellsim subprocess")
        if preexec_fn is not None or startupinfo is not None or creationflags != 0:
            raise ValueError("host process setup options are not supported by shellsim")
        if start_new_session or pass_fds != () or user is not None or group is not None:
            raise ValueError("process identity/session options are not supported by shellsim")
        if extra_groups is not None or umask != -1 or process_group is not None:
            raise ValueError("process identity/session options are not supported by shellsim")
        self._text = bool(text) or bool(universal_newlines) or encoding is not None
        self.encoding = encoding
        self.errors = errors
        self._handle = _shellsim_subprocess.start(args, cwd, env, stdin, stdout, stderr)
        self.pid = self._handle
        self.returncode = None
        self.stdin = _Pipe(self, 0, False) if stdin == PIPE else None
        self.stdout = _Pipe(self, 1, True) if stdout == PIPE else None
        self.stderr = _Pipe(self, 2, True) if stderr == PIPE else None

    def poll(self):
        status = _shellsim_subprocess.poll(self._handle)
        if status is not None:
            self.returncode = status
        return self.returncode

    def wait(self, timeout=None):
        raw = _shellsim_subprocess.wait(self._handle, timeout)
        if raw.timed_out:
            raise TimeoutExpired("Command " + repr(self.args) + " timed out")
        self.returncode = raw.returncode
        return self.returncode

    def communicate(self, input=None, timeout=None):
        if input is None:
            input = b""
        elif self._text:
            encoding = self.encoding if self.encoding is not None else "utf-8"
            errors = self.errors if self.errors is not None else "strict"
            input = input.encode(encoding, errors)
        raw = _shellsim_subprocess.communicate(self._handle, input, timeout)
        if raw.timed_out:
            raise TimeoutExpired("Command " + repr(self.args) + " timed out")
        self.returncode = raw.returncode
        return (
            _decode(raw.stdout, self._text, self.encoding, self.errors),
            _decode(raw.stderr, self._text, self.encoding, self.errors),
        )

    def send_signal(self, signal):
        _shellsim_subprocess.send_signal(self._handle, signal)

    def terminate(self):
        self.send_signal(15)

    def kill(self):
        self.send_signal(9)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.stdin is not None:
            self.stdin.close()
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
        if self.returncode is None:
            self.wait()
