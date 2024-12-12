import dgl

def getGraph(src, dst, nid):
    graph = dgl.graph((src, dst), num_nodes=nid.shape[0])
    graph.ndata['dgl.nid'] = nid
    return graph