"""Exercise a bounded public API inside the actual guest."""

import regex

assert regex.findall(r"\p{Greek}+", "alpha αβγ delta") == ["αβγ"]
assert regex.findall(r"\X", "a\u0301b") == ["a\u0301", "b"]
assert regex.fullmatch(r"(?:cat){e<=1}", "cut").fuzzy_counts == (1, 0, 0)
try:
    regex.compile("[")
except regex.error:
    pass
else:
    raise AssertionError("invalid pattern accepted")
print("regex: public API probe passed")
