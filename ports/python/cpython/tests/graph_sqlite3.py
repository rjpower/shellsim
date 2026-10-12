"""Check file-backed DB-API transactions and reopen through the side extension."""

import sqlite3

path = "/work/transactions.db"
connection = sqlite3.connect(path)
connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, payload BLOB)")
connection.execute("INSERT INTO items VALUES (?, ?, ?)", (1, "committed ' value", b"\x00\xff"))
connection.commit()
connection.execute("INSERT INTO items VALUES (?, ?, ?)", (2, "rolled back", b"discard"))
connection.rollback()
connection.close()

with sqlite3.connect(path) as reopened:
    assert reopened.execute("SELECT id, name, payload FROM items").fetchall() == [(1, "committed ' value", b"\x00\xff")]
    try:
        reopened.execute("INSERT INTO items VALUES (?, ?, ?)", (1, "duplicate", b""))
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("duplicate primary key accepted")
    try:
        reopened.execute("SELECT ?", ())
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("missing SQL parameter accepted")
    reopened.rollback()
reopened.close()
print("sqlite3 parameters, commit, rollback, reopen and DB-API errors passed")
