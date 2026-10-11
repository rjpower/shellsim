"""Exercise local public behavior in the actual guest."""

from shellingham import detect_shell

assert detect_shell(max_depth=3) == ("bash", "bash")

try:
    detect_shell(pid=999999, max_depth=1)
except FileNotFoundError:
    pass
else:
    raise AssertionError("missing virtual process reported a shell")
print("shellingham: guest probe passed")
