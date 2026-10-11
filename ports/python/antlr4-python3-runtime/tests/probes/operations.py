"""Check parser input positioning without host files or subprocesses."""

from antlr4 import InputStream
from antlr4.Token import Token

stream = InputStream("abc")
assert stream.LA(1) == ord("a")
stream.consume()
assert stream.index == 1 and stream.LA(1) == ord("b")
stream.seek(3)
assert stream.LA(1) == Token.EOF
stream.seek(0)
assert stream.getText(0, 2) == "abc"
print("antlr4: input lookahead, seek and EOF passed")
