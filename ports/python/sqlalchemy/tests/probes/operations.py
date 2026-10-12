"""Exercise local public behavior in the actual guest."""

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select
from sqlalchemy.dialects import sqlite

metadata = MetaData()
table = Table("items", metadata, Column("id", Integer, primary_key=True), Column("name", String))
statement = select(table.c.name).where(table.c.id == 7)
compiled = statement.compile(dialect=sqlite.dialect())
assert str(compiled) == "SELECT items.name \nFROM items \nWHERE items.id = ?"
assert compiled.params == {"id_1": 7}

engine = create_engine("sqlite:////work/sqlalchemy.db")
metadata.create_all(engine)
with engine.begin() as connection:
    connection.execute(table.insert(), {"id": 7, "name": "committed ' value"})
with engine.connect() as connection:
    transaction = connection.begin()
    connection.execute(table.insert(), {"id": 8, "name": "rolled back"})
    transaction.rollback()
engine.dispose()
reopened = create_engine("sqlite:////work/sqlalchemy.db")
with reopened.connect() as connection:
    assert connection.execute(select(table.c.id, table.c.name).order_by(table.c.id)).all() == [(7, "committed ' value")]
reopened.dispose()
print("sqlalchemy: guest probe passed")
