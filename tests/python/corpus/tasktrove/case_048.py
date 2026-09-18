from collections import defaultdict
g = defaultdict(list)
g["x"].append(1)
print(g["x"], list(g.keys()))
