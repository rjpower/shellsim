"""Exercise a bounded public API in the actual guest."""

from mock import Mock

callback = Mock(return_value=3)
assert callback("value") == 3
callback.assert_called_once_with("value")
print("mock: guest public API probe passed")
