"""Forward the original Python task entry point to the guest-built C candidate."""

import subprocess

subprocess.run(["/app/solver"], check=True)
