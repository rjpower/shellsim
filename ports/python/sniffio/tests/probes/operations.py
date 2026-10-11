"""Exercise local public behavior in the actual guest."""

import sniffio

try:
    sniffio.current_async_library()
except sniffio.AsyncLibraryNotFoundError:
    pass
else:
    raise AssertionError("unselected async runtime reported")
token = sniffio.current_async_library_cvar.set("asyncio")
try:
    assert sniffio.current_async_library() == "asyncio"
finally:
    sniffio.current_async_library_cvar.reset(token)
print("sniffio: guest probe passed")
