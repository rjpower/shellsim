# NLTK

Qualify corpus-free word/punctuation tokenization, stemming and frequency
counts. Corpus data is not bundled. Network downloading raises
`NotImplementedError` on WASI, whose runtime lacks TLS; other platforms retain
upstream download handlers and security checks. Missing corpora remain explicit.
PanLex imports its optional SQLite backend when its reader is constructed;
SQLite-backed corpora are unqualified in this runtime.
