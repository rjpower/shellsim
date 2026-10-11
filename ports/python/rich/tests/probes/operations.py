"""Render a table and markup through an explicit in-memory output stream."""

import io

from rich.console import Console
from rich.table import Table

stream = io.StringIO()
console = Console(file=stream, width=40, color_system=None)
table = Table("Name", "Value")
table.add_row("answer", "42")
console.print(table)
console.print("[bold]ready[/bold]")
output = stream.getvalue()
assert "Name" in output and "answer" in output and "42" in output and "ready" in output
assert "\x1b" not in output
print("rich: table and markup rendering passed")
