import itertools

import networkx as nx


def create_graph(edges):
    graph = nx.DiGraph()

    for [s, t], latency in edges:
        graph.add_edge(s, t, latency=latency)

    return graph


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def latency(edge, utilization):
    a, b, c = edge

    return a + b * (utilization**c)


def to_edges(route, edges):
    return [edges[(v, w)] for v, w in zip(route[:-1], route[1:])]
