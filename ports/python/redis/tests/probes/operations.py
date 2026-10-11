"""Exercise local public behavior in the actual guest."""

from redis._parsers.encoders import Encoder
from redis.asyncio.connection import SSLConnection
from redis.connection import Connection
from redis.exceptions import RedisError

encoder = Encoder("utf-8", "strict", True)
assert encoder.encode("hello") == b"hello"
assert encoder.decode(b"hello") == "hello"
connection = Connection(host="example.test", port=6379)
assert b"".join(connection.pack_command("SET", "key", "value")) == b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n"
try:
    SSLConnection()
except RedisError:
    pass
else:
    raise AssertionError("unavailable SSL connection accepted")
print("redis: guest probe passed")
