"""Serialize incremental namespaced XML through the stdlib writer."""

from io import BytesIO
from xml.etree.ElementTree import Element, fromstring

from et_xmlfile import xmlfile

output = BytesIO()
with xmlfile(output) as writer:
    with writer.element("{urn:guest}rows", nsmap={None: "urn:guest"}):
        with writer.element("{urn:guest}row", {"label": 'a&"b'}):
            writer.write("<guest> & text")
        leaf = Element("{urn:guest}value")
        leaf.text = "42"
        writer.write(leaf)
root = fromstring(output.getvalue())
assert root.tag == "{urn:guest}rows"
assert root[0].attrib == {"label": 'a&"b'} and root[0].text == "<guest> & text"
assert root[1].tag == "{urn:guest}value" and root[1].text == "42"

print("et-xmlfile: incremental XML passed")
