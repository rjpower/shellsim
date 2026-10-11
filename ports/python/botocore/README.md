# Botocore offline models

The guest probe reads bundled S3 service models and operation shapes. WASI skips
the unavailable TLS imports; attempts to create a TLS context or HTTP session
raise `NotImplementedError`. Host TLS behavior remains unchanged.

AWS connections, credential discovery and request execution are unqualified.
