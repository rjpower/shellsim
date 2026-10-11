"""Exercise a bounded public API inside the actual guest."""

from defusedxml.common import EntitiesForbidden
from defusedxml.ElementTree import fromstring

assert fromstring("<root><value>ok</value></root>").find("value").text == "ok"
try:
    fromstring('<!DOCTYPE root [<!ENTITY payload "expanded">]><root>&payload;</root>')
except EntitiesForbidden:
    pass
else:
    raise AssertionError("entity expansion accepted")
print("defusedxml: public API probe passed")
