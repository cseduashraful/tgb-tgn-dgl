import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl.function as fn

class TGNN(nn.Module):
    def __init__(self, ef_dim, hidden_dim):
        super().__init__()
        
        self.timeEncode = nn.Linear(1, hidden_dim)
        self.timedEdge = nn.Linear(ef_dim+hidden_dim, hidden_dim)
        self.predictor = EdgePredictor(hidden_dim)
        self.reset_parameters()


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.timeEncode.weight, gain=gain)
        nn.init.xavier_normal_(self.timedEdge.weight, gain=gain)

    def forward(self, g, ef, bt, blocks, neg_samples=1):
        bt = bt.view(-1, 1)
        s, p, n, t, m, assoc, new_assoc = blocks
        # new_assoc = assoc.clone()
        s_emb = []
        p_emb = []
        n_emb = []
        for idx, tidx in enumerate(t):
            # print("edges: ", g.num_edges())

            # print("p[idx] shape: ", p[idx].shape)
            # print("n[idx] shape: ", n[idx].shape)

            pos_root_nodes  =  torch.cat([s[idx], p[idx]], dim=0).unique()
            root_nodes = torch.cat([pos_root_nodes, n[idx].view(-1)], dim=0).unique()
            # print(s[idx])
            # print(root_nodes)
            # print(new_assoc)
            # print(assoc)
            
            mapped_nids = assoc[root_nodes]
            # print(new_assoc[root_nodes])
            new_assoc[root_nodes] = torch.arange(len(root_nodes), device = new_assoc.device)

            subg = dgl.node_subgraph(g, mapped_nids)
            subg_ef = ef[subg.edata['_ID']]
            subg_bt = bt[subg.edata['_ID']]

            # print("sidx: ", s[idx])
            # print("pidx: ", p[idx])
            # print("new index: ",new_assoc[p[idx]])
            # print("nidx: ", n[idx])
            # print("subg ndata: ", subg.ndata)
            # print("subg edata: ", subg.edata)
            
            # print("subg_bt: ", subg_bt)
            # input("continue: ")

            # print("31: ", torch.cuda.memory_allocated())
            with subg.local_scope():
                tf = F.relu(self.timeEncode(subg_bt))
                subg.edata['cf'] = F.relu(self.timedEdge(torch.cat((subg_ef, tf), dim=1)))
                subg.update_all(message_func=dgl.function.copy_e('cf', 'msg'),
                reduce_func=dgl.function.sum('msg', 'h'))
                # return g.ndata['h']
                s_emb.append(subg.ndata['h'][new_assoc[s[idx]]])
                p_emb.append(subg.ndata['h'][new_assoc[p[idx]]])
                n_emb.append(subg.ndata['h'][new_assoc[n[idx].view(-1)]])
                # print("41: ", torch.cuda.memory_allocated())
            #update graph (add positive edges with features m)
            # print("43: ", torch.cuda.memory_allocated()) 
                
            g.add_edges(assoc[s[idx]], assoc[p[idx]])
            g.add_edges(assoc[p[idx]], assoc[s[idx]])
            ef = torch.cat([ef, m[idx], m[idx]], dim=0)
            bt = torch.cat([bt, tidx.view(-1, 1), tidx.view(-1, 1)], dim=0)

        return self.predictor(torch.cat(s_emb, dim=0), torch.cat(p_emb, dim=0), torch.cat(n_emb, dim=0), neg_samples=neg_samples)

"""
Edge predictor from Graph Mixer
"""

class EdgePredictor(torch.nn.Module):
    """
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """
    def __init__(self, dim_in_node):
        super().__init__()

        # self.dim_in_time = dim_in_time
        # self.dim_in_node = dim_in_node

        self.src_fc = torch.nn.Linear(dim_in_node, dim_in_node)
        self.dst_fc = torch.nn.Linear(dim_in_node, dim_in_node)
        self.out_fc = torch.nn.Linear(dim_in_node, 1)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h_src, h_pos_dst, h_neg_dst, neg_samples=1):
        # num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h_src)
        h_pos_dst = self.dst_fc(h_pos_dst)
        h_neg_dst = self.dst_fc(h_neg_dst)
        h_pos_edge = F.relu(h_src + h_pos_dst)
        h_neg_edge = F.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        # h_pos_edge = torch.nn.functional.relu(h_pos_dst)
        # h_neg_edge = torch.nn.functional.relu(h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


def getModel(feature_dim, hidden_dim, device):
    gnn = TGNN(feature_dim, hidden_dim).to(device)
    # pred = MLPPredictor(hidden_dim).to(device)
    return {"gnn": gnn}