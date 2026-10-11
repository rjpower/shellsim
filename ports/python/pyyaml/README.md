# PyYAML

Build PyYAML 6.0.3 with its upstream pure Python parser. The patch omits the
optional LibYAML extension and its Cython build requirement; Python package
sources are unchanged. Safe loading, dumping, and malformed-input rejection
are covered by the guest probe. `CLoader` and `CDumper` are outside this product.
