"""Exercise a bounded public API inside the actual guest."""

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

assert wordpunct_tokenize("A small test, twice!") == ["A", "small", "test", ",", "twice", "!"]
assert PorterStemmer().stem("running") == "run"
assert nltk.FreqDist(["a", "b", "a"]).most_common(1) == [("a", 2)]
try:
    nltk.data.find("corpora/missing_buildomatic_probe")
except LookupError:
    pass
else:
    raise AssertionError("missing corpus accepted")
try:
    nltk.pathsec.urlopen("https://example.invalid/corpus")
except NotImplementedError:
    pass
else:
    raise AssertionError("network downloader accepted")
print("nltk: tokenization, stemming, counts and corpus/download frontiers passed")
