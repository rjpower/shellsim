"""Generate bounded examples without deadlines or a host database."""

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument


@settings(max_examples=20, derandomize=True, deadline=None, database=None)
@given(st.lists(st.integers(-10, 10), max_size=8))
def check_sum(values):
    assert sum(values) == sum(reversed(values))


check_sum()
try:
    st.integers(min_value=2, max_value=1).validate()
except InvalidArgument:
    pass
else:
    raise AssertionError("invalid integer range accepted")
print("hypothesis: deterministic examples and invalid strategy passed")
