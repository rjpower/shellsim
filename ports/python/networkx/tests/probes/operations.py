"""Exercise a bounded public API in the actual guest."""

import networkx as nx

graph = nx.DiGraph([(0, 1), (1, 2), (0, 3)])
assert nx.shortest_path(graph, 0, 2) == [0, 1, 2]
assert nx.is_directed_acyclic_graph(graph)
nx.write_edgelist(graph, "/work/graph.bz2", data=False)
restored = nx.read_edgelist("/work/graph.bz2", create_using=nx.DiGraph, nodetype=int)
assert set(restored.edges) == set(graph.edges)
graph.add_edge(2, 0)
assert not nx.is_directed_acyclic_graph(graph)
print("networkx: guest public API probe passed")
