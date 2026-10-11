"""Exercise lexer lookup and ordinary syntax highlighting in the guest."""

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

output = highlight("answer = 42\n", get_lexer_by_name("python"), HtmlFormatter())
assert "answer" in output and "42" in output and "<span" in output
try:
    get_lexer_by_name("missing-shellsim-lexer")
except ClassNotFound:
    pass
else:
    raise AssertionError("unknown lexer accepted")
print("pygments: highlighting and unknown lexer passed")
