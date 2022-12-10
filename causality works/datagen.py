import networkx as nx
import numpy as np

class DataGenerator():
    def __init__(self, causalgraph, nnodes):
        self.set_graph(causalgraph, nnodes)
        self.sample_funcs = {}
        self.child_funcs = {}

    def _default_child_func(self, data, preds):
        val = 0
        for edge in preds:
            val += data[edge[0]-1] * edge[2]["weight"]
        return val + np.random.normal()*0.05

    def set_graph(self, graph, nnodes):
        self.causalgraph = graph
        self.topsort = list(nx.topological_sort(graph))
        self.nnodes = nnodes

    def set_sample_func(self, node, f):
        self.sample_funcs[node] = f

    def set_child_func(self, node, f):
        self.child_funcs[node] = f

    def generate(self, N):
        data = np.zeros((N, self.nnodes))
        for i in range(N):
            for j in self.topsort:
                preds = list(self.causalgraph.in_edges(j, data=True))
                if len(preds) == 0:
                    func = np.random.normal if not j in self.sample_funcs else self.sample_funcs[j]
                    data[i][j-1] = func()
                else:
                    func = self._default_child_func if not j in self.child_funcs else self.child_funcs[j]
                    data[i][j-1] = func(data[i], preds)
        return data
