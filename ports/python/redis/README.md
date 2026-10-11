# Redis protocol helpers

The guest probe encodes values and packs a Redis command without opening a
connection. Two optional SSL type annotations are postponed so upstream's
existing missing-SSL error can run; SSL construction still raises `RedisError`.

Redis service access and asynchronous connection execution are unqualified.
