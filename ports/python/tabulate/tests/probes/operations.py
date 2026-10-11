"""Format a small table without terminal or locale dependencies."""

from tabulate import tabulate

output = tabulate([["answer", 42]], headers=["Name", "Value"], tablefmt="grid")
assert "Name" in output and "Value" in output and "answer" in output and "42" in output
assert tabulate([], tablefmt="plain") == ""
print("tabulate: table and empty input passed")
