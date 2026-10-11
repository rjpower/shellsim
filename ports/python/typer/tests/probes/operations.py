"""Exercise local public behavior in the actual guest."""

import typer
from typer.testing import CliRunner

app = typer.Typer()


@app.command()
def greet(name: str):
    typer.echo("Hello " + name)


result = CliRunner().invoke(app, ["guest"])
assert result.exit_code == 0, result.output
assert "Hello guest" in result.output
print("typer: guest probe passed")
