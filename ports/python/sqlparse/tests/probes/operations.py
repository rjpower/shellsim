"""Exercise a bounded public API in the actual guest."""

import sqlparse

parsed = sqlparse.parse("select name from things where id = 3;")
assert len(parsed) == 1
assert parsed[0].get_type() == "SELECT"
assert "SELECT" in sqlparse.format("select name from things", keyword_case="upper")
print("sqlparse: guest public API probe passed")
