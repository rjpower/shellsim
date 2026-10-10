from zss import Node, simple_distance

a = Node("root").addkid(Node("left")).addkid(Node("right"))
b = Node("root").addkid(Node("left")).addkid(Node("changed"))
assert simple_distance(a, a) == 0
assert simple_distance(a, b) == 1
assert simple_distance(Node("a"), Node("b")) == 1
print("Zss tree edit distances passed")
