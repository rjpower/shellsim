import dataclasses
@dataclasses.dataclass
class Point:
    x: int
print(Point(2).x)
