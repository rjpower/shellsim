"""Exercise local public behavior in the actual guest."""

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 17
assert detect("This is an ordinary English sentence about people and their daily lives.") == "en"
print("langdetect: guest probe passed")
