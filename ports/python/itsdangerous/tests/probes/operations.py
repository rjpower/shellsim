"""Exercise a bounded public API in the actual guest."""

from itsdangerous import BadSignature, URLSafeSerializer

serializer = URLSafeSerializer("fixed-test-key")
token = serializer.dumps({"count": 3})
assert serializer.loads(token) == {"count": 3}
try:
    serializer.loads("broken." + token)
except BadSignature:
    pass
else:
    raise AssertionError("invalid signature accepted")
print("itsdangerous: guest public API probe passed")
