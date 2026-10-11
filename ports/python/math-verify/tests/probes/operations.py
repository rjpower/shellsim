"""Smoke fixed math expressions; this does not qualify alarm-based grading."""

from math_verify import parse, verify

gold = parse(r"$\frac{1}{2}$", parsing_timeout=0)
equal = parse(r"$0.5$", parsing_timeout=0)
different = parse(r"$0.6$", parsing_timeout=0)
assert gold and equal and different
assert verify(gold, equal, timeout_seconds=0)
assert not verify(gold, different, timeout_seconds=0)
print("math-verify: fixed expressions passed; alarm-based grading remains unqualified")
