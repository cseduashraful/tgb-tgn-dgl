import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl.function as fn

class TGNN(nn.Module):
    def __init__(self, ef_dim, hidden_dim, num_nodes, device):
        super().__init__()

        time_dim = hidden_dim

        self.time_assoc = torch.zeros(num_nodes, dtype=torch.float, requires_grad=False).to(device)
        self.num_nodes = num_nodes
        self.memory = MemoryModule(self.num_nodes, hidden_dim)
        time_encode = TimeEncode(time_dim)
        self.temporal_edge_process = TemporalEdgePreprocess(time_encode)

    


        self.assoc = torch.empty(num_nodes, dtype=torch.long)
        # self.timeEncode = TimeEncode(hidden_dim)#nn.Linear(1, hidden_dim)
        self.timedEdge = nn.Linear(ef_dim+hidden_dim, hidden_dim)

        self.predictor = EdgePredictor(ef_dim+time_dim, hidden_dim)        
        self.reset_parameters()


    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_normal_(self.timeEncode.weight, gain=gain)
        nn.init.xavier_normal_(self.timedEdge.weight, gain=gain)
    

    

    def forward(self, g, ef, bt, blocks, neg_samples=1):
        bt = bt.view(-1, 1) #time associated with each edge
        s, p, n, t, m, assoc = blocks
        # new_assoc = assoc.clone()
        s_emb = []
        p_emb = []
        n_emb = []
        for idx, tidx in enumerate(t):
            # print("edges: ", g.num_edges())

            # print("p[idx] shape: ", p[idx].shape)
            # print("n[idx] shape: ", n[idx].shape)

            # print(self.time_assoc)
            # print(n[idx])
            # print(tidx)
            if len(n[idx].shape) > 1:
                mx = tidx.max()
                self.time_assoc[:] = mx
            else:
                self.time_assoc[n[idx]] = tidx
            self.time_assoc[p[idx]] = tidx
            self.time_assoc[s[idx]] = tidx

            pos_root_nodes  =  torch.cat([s[idx], p[idx]], dim=0).unique()
            root_nodes = torch.cat([pos_root_nodes, n[idx].view(-1)], dim=0).unique()
            # print(s[idx])
            
            # print(new_assoc)
            # print(assoc)
            
            mapped_nids = assoc[root_nodes]
           
            
            #DGL Specific

            # subg = dgl.node_subgraph(g, mapped_nids)
            subg = dgl.in_subgraph(g, mapped_nids)
            self.assoc[subg.ndata['dgl.nid']] = torch.arange(len(subg.ndata['dgl.nid']), device = self.assoc.device)
            subg_ef = ef[subg.edata['_ID'], :]
            subg_bt = bt[subg.edata['_ID'], :]
            # print("subg bt shape: ", subg_bt.shape)
            subg_node_memory = self.memory.memory[subg.ndata['dgl.nid'], :]

            # print("sidx: ", s[idx])
            # print("pidx: ", p[idx])
            # print("new index: ",new_assoc[p[idx]])
            # print("nidx: ", n[idx])
            
            # print("subg edata: ", subg.edata)
            
            # print("subg_bt: ", subg_bt)
            # if subg.num_nodes() != root_nodes.shape[0]:
            #     print("root nodes: ", root_nodes)
            #     print("subg ndata: ", subg.ndata['dgl.nid'])
            #     print(new_assoc[root_nodes])

            #     input("continue: ")

            # print("31: ", torch.cuda.memory_allocated())
            with subg.local_scope():
                # tf = F.relu(self.timeEncode(subg_bt))
                # subg.edata['cf'] = F.relu(self.timedEdge(torch.cat((subg_ef, tf), dim=1)))
                subg.ndata['timestamp'] = self.time_assoc[subg.ndata['dgl.nid']].view(-1,1)
                subg.edata['timestamp'] = subg_bt
                # print("subg.ndata['timestamp'] ", subg.ndata['timestamp'].shape)
                # print("subg.edata['timestamp'] ", subg.edata['timestamp'].shape)
                subg.edata['feats'] = subg_ef
                subg.edata['cf'] = self.temporal_edge_process(subg)

                subg.update_all(message_func=dgl.function.copy_e('cf', 'msg'),
                reduce_func=dgl.function.sum('msg', 'h'))
                # return g.ndata['h']
                s_emb.append(subg.ndata['h'][self.assoc[s[idx]]])
                p_emb.append(subg.ndata['h'][self.assoc[p[idx]]])
                n_emb.append(subg.ndata['h'][self.assoc[n[idx].view(-1)]])
                # print("41: ", torch.cuda.memory_allocated())
            #update graph (add positive edges with features m)
            # print("43: ", torch.cuda.memory_allocated()) 
                
            g.add_edges(assoc[s[idx]], assoc[p[idx]])
            g.add_edges(assoc[p[idx]], assoc[s[idx]])

            #End of DGL specific

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
    def __init__(self, dim_in_node, dim_out):
        super().__init__()

        # self.dim_in_time = dim_in_time
        # self.dim_in_node = dim_in_node

        self.src_fc = torch.nn.Linear(dim_in_node, dim_out)
        self.dst_fc = torch.nn.Linear(dim_in_node, dim_out)
        self.out_fc = torch.nn.Linear(dim_out, 1)
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





class TimeEncode(nn.Module):
    """Use finite fourier series with different phase and frequency to encode
    time different between two event

    ..math::
        \Phi(t) = [\cos(\omega_0t+\psi_0),\cos(\omega_1t+\psi_1),...,\cos(\omega_nt+\psi_n)] 

    Parameter
    ----------
    dimension : int
        Length of the fourier series. The longer it is , 
        the more timescale information it can capture

    Example
    ----------
    >>> tecd = TimeEncode(10)
    >>> t = torch.tensor([[1]])
    >>> tecd(t)
    tensor([[[0.5403, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000]]], dtype=torch.float64, grad_fn=<CosBackward>)
    """

    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # print("t shape: ", t.shape)
        # t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        # print(output.shape)
        return output


class MemoryModule(nn.Module):
    """Memory module as well as update interface

    The memory module stores both historical representation in last_update_t

    Parameters
    ----------
    n_node : int
        number of node of the entire graph

    hidden_dim : int
        dimension of memory of each node

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py;
                     examples/pytorch/tgn/train.py 

    """

    def __init__(self, n_node, hidden_dim, mem_device=None):
        super(MemoryModule, self).__init__()
        self.n_node = n_node
        self.hidden_dim = hidden_dim
        self.mem_device = mem_device
        self.create_memory()
        
    def create_memory(self):  
        self.last_update_t = nn.Parameter(torch.zeros(
            self.n_node).float(), requires_grad=False)
        self.memory = nn.Parameter(torch.zeros(
            (self.n_node, self.hidden_dim)).float(), requires_grad=False)
        
    def reset_memory(self):  
        # self.last_update_t = nn.Parameter(torch.zeros(
        #     self.n_node).float(), requires_grad=False)
        # self.memory = nn.Parameter(torch.zeros(
        #     (self.n_node, self.hidden_dim)).float(), requires_grad=False)
        # print(self.memory)
        if self.mem_device is not None:
            self.last_update_t = nn.Parameter(torch.zeros(
                self.n_node, device=self.mem_device).float(), requires_grad=False)
            self.memory = nn.Parameter(torch.zeros(
                (self.n_node, self.hidden_dim), device=self.mem_device).float(), requires_grad=False)
        else:
            self.last_update_t = nn.Parameter(torch.zeros(
                self.n_node).float(), requires_grad=False)
            self.memory = nn.Parameter(torch.zeros(
                (self.n_node, self.hidden_dim)).float(), requires_grad=False)

    def backup_memory(self):
        """
        Return a deep copy of memory state and last_update_t
        For test new node, since new node need to use memory upto validation set
        After validation, memory need to be backed up before run test set without new node
        so finally, we can use backup memory to update the new node test set
        """
        return self.memory.clone(), self.last_update_t.clone()

    def restore_memory(self, memory_backup):
        """Restore the memory from validation set

        Parameters
        ----------
        memory_backup : (memory,last_update_t)
            restore memory based on input tuple
        """
        self.memory = memory_backup[0].clone()
        self.last_update_t = memory_backup[1].clone()

    # Which is used for attach to subgraph
    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    # When the memory need to be updated
    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def set_last_update_t(self, node_idxs, values):
        self.last_update_t[node_idxs] = values

    # For safety check
    def get_last_update(self, node_idxs):
        return self.last_update_t[node_idxs]

    def detach_memory(self):
        """
        Disconnect the memory from computation graph to prevent gradient be propagated multiple
        times
        """
        self.memory.detach_()


class MemoryOperation(nn.Module):
    """ Memory update using message passing manner, update memory based on positive
    pair graph of each batch with recurrent module GRU or RNN

    Message function
    ..math::
        m_i(t) = concat(memory_i(t^-),TimeEncode(t),v_i(t))

    v_i is node feature at current time stamp

    Aggregation function
    ..math::
        \bar{m}_i(t) = last(m_i(t_1),...,m_i(t_b))

    Update function
    ..math::
        memory_i(t) = GRU(\bar{m}_i(t),memory_i(t-1))

    Parameters
    ----------

    updater_type : str
        indicator string to specify updater

        'rnn' : use Vanilla RNN as updater

        'gru' : use GRU as updater

    memory : MemoryModule
        memory content for update

    e_feat_dim : int
        dimension of edge feature

    temporal_dim : int
        length of fourier series for time encoding

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py
    """

    def __init__(self, updater_type, memory, e_feat_dim, temporal_encoder):
        super(MemoryOperation, self).__init__()
        updater_dict = {'gru': nn.GRUCell, 'rnn': nn.RNNCell}
        self.memory = memory
        memory_dim = self.memory.hidden_dim
        self.temporal_encoder = temporal_encoder
        self.message_dim = memory_dim+memory_dim + \
            e_feat_dim+self.temporal_encoder.dimension
        self.updater = updater_dict[updater_type](input_size=self.message_dim,
                                                  hidden_size=memory_dim)
        self.memory = memory

    # Here assume g is a subgraph from each iteration
    def stick_feat_to_graph(self, g):
        # How can I ensure order of the node ID
        g.ndata['timestamp'] = self.memory.last_update_t[g.ndata[dgl.NID]]
        g.ndata['memory'] = self.memory.memory[g.ndata[dgl.NID]]

    def msg_fn_cat(self, edges):
        src_delta_time = edges.data['timestamp'] - edges.src['timestamp']
        time_encode = self.temporal_encoder(src_delta_time.unsqueeze(
            dim=1)).view(len(edges.data['timestamp']), -1)
        ret = torch.cat([edges.src['memory'], edges.dst['memory'],
                         edges.data['feats'], time_encode], dim=1)
        return {'message': ret, 'timestamp': edges.data['timestamp']}

    def agg_last(self, nodes):
        timestamp, latest_idx = torch.max(nodes.mailbox['timestamp'], dim=1)
        ret = nodes.mailbox['message'].gather(1, latest_idx.repeat(
            self.message_dim).view(-1, 1, self.message_dim)).view(-1, self.message_dim)
        return {'message_bar': ret.reshape(-1, self.message_dim), 'timestamp': timestamp}

    def update_memory(self, nodes):
        # It should pass the feature through RNN
        ret = self.updater(
            nodes.data['message_bar'].float(), nodes.data['memory'].float())
        return {'memory': ret}

    def forward(self, g):
        self.stick_feat_to_graph(g)
        g.update_all(self.msg_fn_cat, self.agg_last, self.update_memory)
        return g





class TemporalEdgePreprocess(nn.Module):
    '''Preprocess layer, which finish time encoding and concatenate 
    the time encoding to edge feature.

    Parameter
    ==========
    edge_feats : int
        number of orginal edge feature

    temporal_encoder : torch.nn.Module
        time encoder model
    '''

    def __init__(self, temporal_encoder):
        super(TemporalEdgePreprocess, self).__init__()
        # self.edge_feats = edge_feats
        self.temporal_encoder = temporal_encoder

    def edge_fn(self, edges):
        t0 = torch.zeros_like(edges.dst['timestamp'])
        time_diff = edges.data['timestamp'] - edges.src['timestamp']
        # print("t0: ", t0.shape)
        # print("time_diff: ", time_diff.shape)
        # print("edges.data['timestamp'] ", edges.data['timestamp'].shape)
        # print("edges.src['timestamp'] ", edges.src['timestamp'].shape)
        time_encode = self.temporal_encoder(time_diff)  #self.temporal_encoder(time_diff.unsqueeze(dim=1)).view(t0.shape[0], -1)
        edge_feat = torch.cat([edges.data['feats'], time_encode], dim=1)
        return {'efeat': edge_feat}

    def forward(self, graph):
        # print(graph.num_edges())
        graph.apply_edges(self.edge_fn)
        efeat = graph.edata['efeat']
        return efeat





def getModel(feature_dim, hidden_dim, num_nodes, device):
    gnn = TGNN(feature_dim, hidden_dim, num_nodes, device).to(device)
    # pred = MLPPredictor(hidden_dim).to(device)
    return {"gnn": gnn}