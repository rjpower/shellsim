"""Load bundled font data and reject an absent font."""

from pyfiglet import Figlet, FontNotFound

figure = Figlet(font="standard")
assert figure.renderText("A").strip()
assert figure.renderText("A") == figure.renderText("A")
try:
    Figlet(font="missing-shellsim-font")
except FontNotFound:
    pass
else:
    raise AssertionError("absent font accepted")
print("pyfiglet: bundled font rendering and absent font passed")
