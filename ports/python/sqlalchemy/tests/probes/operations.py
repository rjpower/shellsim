"""Exercise local public behavior in the actual guest."""

from sqlalchemy import Column, Integer, MetaData, String, Table, select
from sqlalchemy.dialects import sqlite

metadata = MetaData()
table = Table("items", metadata, Column("id", Integer, primary_key=True), Column("name", String))
statement = select(table.c.name).where(table.c.id == 7)
compiled = statement.compile(dialect=sqlite.dialect())
assert str(compiled) == "SELECT items.name \nFROM items \nWHERE items.id = ?"
assert compiled.params == {"id_1": 7}
print("sqlalchemy: guest probe passed")
