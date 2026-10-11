"""Exercise a bounded public API in the actual guest."""

from typing import Annotated

from typing_inspection.introspection import AnnotationSource, inspect_annotation

result = inspect_annotation(Annotated[int, "bounded"], annotation_source=AnnotationSource.CLASS)
assert result.type is int
assert result.metadata == ["bounded"]
print("typing-inspection: guest public API probe passed")
