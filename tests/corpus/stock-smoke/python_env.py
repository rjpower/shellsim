import os
from pathlib import Path

print(os.environ["HOME"])
print(Path("input/message.txt").read_text(), end="")
