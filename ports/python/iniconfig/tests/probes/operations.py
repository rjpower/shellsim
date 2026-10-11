"""Parse named sections and reject malformed configuration in the guest."""

from iniconfig import IniConfig, ParseError

config = IniConfig("settings.ini", data="[build]\njobs = 32\n[cache]\nenabled = yes\n")
assert config["build"]["jobs"] == "32"
assert config["cache"]["enabled"] == "yes"
try:
    IniConfig("broken.ini", data="value outside section")
except ParseError:
    pass
else:
    raise AssertionError("malformed INI accepted")
print("iniconfig: sections and malformed input passed")
