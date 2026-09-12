"""Synchronous subprocess compatibility over shellsim logical processes."""

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
    if input is None:
        input = b""
    elif _text_mode(text, encoding):
        if encoding is None:
            encoding = "utf-8"
        if errors is None:
            errors = "strict"
        input = input.encode(encoding, errors)

    raw = _shellsim_subprocess.run(args, input, cwd, env, timeout, stdout, stderr)
    child_stdout = _decode(raw.stdout, text, encoding, errors)
    child_stderr = _decode(raw.stderr, text, encoding, errors)
    if raw.timed_out:
        raise TimeoutExpired("Command " + repr(args) + " timed out")
    result = CompletedProcess(original_args, raw.returncode, child_stdout, child_stderr)
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


class Popen:
    def __init__(self, *args):
        raise RuntimeError("Popen requires live asynchronous processes and is not supported")
