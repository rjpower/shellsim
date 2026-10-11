"""Exercise CLI parsing through its in-process testing runner."""

import click
from click.testing import CliRunner


@click.command()
@click.option("--count", type=click.IntRange(1, 3), default=1)
def command(count):
    click.echo("hi " * count)


runner = CliRunner()
assert runner.invoke(command, ["--count", "2"]).output == "hi hi \n"
assert runner.invoke(command, ["--count", "0"]).exit_code == 2
print("click: option parsing, output and rejection passed")
