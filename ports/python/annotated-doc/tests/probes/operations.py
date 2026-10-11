"""Exercise local public behavior in the actual guest."""

from typing import Annotated, get_args

from annotated_doc import Doc

annotation = Annotated[int, Doc("A documented count")]
assert get_args(annotation)[0] is int
assert get_args(annotation)[1].documentation == "A documented count"
print("annotated-doc: guest probe passed")
