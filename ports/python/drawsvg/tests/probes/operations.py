"""Serialize a vector drawing without optional raster dependencies."""

import xml.etree.ElementTree as ET

import drawsvg as draw

drawing = draw.Drawing(20, 10)
drawing.append(draw.Rectangle(0, 0, 10, 5, fill="red"))
root = ET.fromstring(drawing.as_svg())
rectangles = root.findall("{http://www.w3.org/2000/svg}rect")
assert len(rectangles) == 1
assert rectangles[0].attrib["fill"] == "red"
assert float(rectangles[0].attrib["width"]) == 10
print("drawsvg: vector serialization passed")
