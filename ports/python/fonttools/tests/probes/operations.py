"""Exercise the pure OpenType utilities used by plotting."""

from fontTools.misc.fixedTools import fixedToFloat, floatToFixed
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont

assert fixedToFloat(floatToFixed(1.5, 16), 16) == 1.5
pen = RecordingPen()
pen.moveTo((0, 0))
pen.lineTo((1, 2))
pen.closePath()
assert pen.value == [("moveTo", ((0, 0),)), ("lineTo", ((1, 2),)), ("closePath", ())]
font = TTFont()
assert "head" not in font
try:
    font["head"]
except KeyError:
    pass
else:
    raise AssertionError("missing font table accepted")
print("fonttools: fixed-point, outlines and absent table passed")
