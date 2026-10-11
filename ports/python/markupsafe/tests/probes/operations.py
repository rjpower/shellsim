"""Exercise local public behavior in the actual guest."""

from markupsafe import Markup, escape
from markupsafe._speedups import _escape_inner

assert escape("<x>&") == "&lt;x&gt;&amp;"
assert _escape_inner("<x>&") == "&lt;x&gt;&amp;"
assert str(Markup("<b>%s</b>") % "<x>") == "<b>&lt;x&gt;</b>"
print("markupsafe: guest probe passed")
