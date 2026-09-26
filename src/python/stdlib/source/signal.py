"""Signal numbers accepted by `os.kill`, matching shellsim's modeled signal set.

Only the signals shellsim's process layer models are exposed. Sending any other value raises
ValueError from `os.kill`, matching CPython's behavior for an unsupported signal number.
"""

SIGHUP = 1
SIGINT = 2
SIGKILL = 9
SIGUSR1 = 10
SIGUSR2 = 12
SIGPIPE = 13
SIGTERM = 15
