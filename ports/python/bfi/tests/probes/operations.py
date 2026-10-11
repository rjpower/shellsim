"""Run bounded Brainfuck programs and reject unmatched brackets."""

import bfi

assert bfi.interpret("++++++++[>++++++++<-]>+.", buffer_output=True) == "A"
assert bfi.interpret(",+.", input_data="A", buffer_output=True) == "B"
try:
    bfi.parse("[")
except bfi.BrainfuckSyntaxError:
    pass
else:
    raise AssertionError("unmatched bracket accepted")
print("bfi: loops, input and invalid syntax passed")
