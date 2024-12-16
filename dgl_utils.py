import dgl

def getGraph(src, dst, nid, self_loop = False):
    graph = dgl.graph((src, dst), num_nodes=nid.shape[0])
    graph.ndata['dgl.nid'] = nid
    if self_loop:
        graph = dgl.add_self_loop(graph)
    return graph