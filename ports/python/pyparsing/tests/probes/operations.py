"""Parse arithmetic tokens and preserve syntax rejection."""

import pyparsing as pp

grammar = pp.pyparsing_common.integer + pp.Literal("+").suppress() + pp.pyparsing_common.integer
assert grammar.parse_string("12 + 30", parse_all=True).as_list() == [12, 30]
try:
    grammar.parse_string("12 + x", parse_all=True)
except pp.ParseException:
    pass
else:
    raise AssertionError("malformed arithmetic accepted")
print("pyparsing: token parsing and invalid syntax passed")
