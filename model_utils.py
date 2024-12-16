
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp

import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
from dgl.base import DGLError

class TGNN(nn.Module):
    def __init__(self, ef_dim, hidden_dim, num_nodes, device, num_heads = 8, layers = 1, time_dim = 100):
        super().__init__()

        time_dim = hidden_dim
        self.memory_dim = hidden_dim


        self.time_assoc = torch.zeros(num_nodes, dtype=torch.float, requires_grad=False).to(device)
        self.num_nodes = num_nodes
        self.edge_feat_dim = ef_dim
        self.embedding_dim = hidden_dim
        self.num_heads = num_heads
        self.layers = layers



        self.memory = MemoryModule(self.num_nodes, hidden_dim)
        self.temporal_encoder = TimeEncode(time_dim)
        # self.temporal_edge_process = TemporalEdgePreprocess(self.temporal_encoder)
        self.embedding_attn = TemporalTransformerConv(self.edge_feat_dim,
                                                      self.memory_dim,
                                                      self.temporal_encoder,
                                                      self.embedding_dim,
                                                      self.num_heads,
                                                      layers=self.layers,
                                                      allow_zero_in_degree=True)

    


        self.assoc = torch.empty(num_nodes, dtype=torch.long)
        # self.timeEncode = TimeEncode(hidden_dim)#nn.Linear(1, hidden_dim)
        # self.timedEdge = nn.Linear(ef_dim+hidden_dim, hidden_dim)

        self.predictor = EdgePredictor(hidden_dim, hidden_dim)        
        # self.reset_parameters()


    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain('relu')
    #     # nn.init.xavier_normal_(self.timeEncode.weight, gain=gain)
    #     nn.init.xavier_normal_(self.timedEdge.weight, gain=gain)
    

    

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
            # print("g edata: ", g.num_edges() )
            subg = dgl.in_subgraph(g, mapped_nids)
            # print("subg.edata['_ID']: ", subg.edata['_ID'])
            # input("subg.edata['_ID']: ")
            self.assoc[subg.ndata['dgl.nid']] = torch.arange(len(subg.ndata['dgl.nid']), device = self.assoc.device)
            subg_ef = ef[subg.edata['_ID'], :]
            subg_bt = bt[subg.edata['_ID'], :]
            # print("subg bt shape: ", subg_bt.shape)
            subg_node_memory = self.memory.memory[subg.ndata['dgl.nid'], :]
            # emb_memory = self.memory.memory[emb_graph.ndata[dgl.NID], :]

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

                embed  =  self.embedding_attn(subg, subg_node_memory)
                s_emb.append(embed[self.assoc[s[idx]]])
                p_emb.append(embed[self.assoc[p[idx]]])
                n_emb.append(embed[self.assoc[n[idx].view(-1)]])

                # subg.edata['cf'] = self.temporal_edge_process(subg)

                # subg.update_all(message_func=dgl.function.copy_e('cf', 'msg'),
                # reduce_func=dgl.function.sum('msg', 'h'))
                # # return g.ndata['h']
                # s_emb.append(subg.ndata['h'][self.assoc[s[idx]]])
                # p_emb.append(subg.ndata['h'][self.assoc[p[idx]]])
                # n_emb.append(subg.ndata['h'][self.assoc[n[idx].view(-1)]])
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
        self.memory = nn.Parameter(torch.ones(
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


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x
    
class EdgeGATConv(nn.Module):
    '''Edge Graph attention compute the graph attention from node and edge feature then aggregate both node and
    edge feature.

    Parameter
    ==========
    node_feats : int
        number of node features

    edge_feats : int
        number of edge features

    out_feats : int
        number of output features

    num_heads : int
        number of heads in multihead attention

    feat_drop : float, optional
        drop out rate on the feature

    attn_drop : float, optional
        drop out rate on the attention weight

    negative_slope : float, optional
        LeakyReLU angle of negative slope.

    residual : bool, optional
        whether use residual connection

    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.

    '''

    def __init__(self,
                 node_feats,
                 edge_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(EdgeGATConv, self).__init__()
        self._num_heads = num_heads
        self._node_feats = node_feats
        self._edge_feats = edge_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.fc_node = nn.Linear(
            self._node_feats, self._out_feats*self._num_heads)
        self.fc_edge = nn.Linear(
            self._edge_feats, self._out_feats*self._num_heads)
        self.attn_l = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(
            size=(1, self._num_heads, self._out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.residual = residual
        if residual:
            if self._node_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._node_feats, self._out_feats*self._num_heads, bias=False)
            else:
                self.res_fc = Identity()
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_node.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if self.residual and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def msg_fn(self, edges):
        ret = edges.data['a'].view(-1, self._num_heads,
                                   1)*edges.data['el_prime']
        return {'m': ret}

    def forward(self, graph, nfeat, efeat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            nfeat = self.feat_drop(nfeat)
            efeat = self.feat_drop(efeat)

            node_feat = self.fc_node(
                nfeat).view(-1, self._num_heads, self._out_feats)
            edge_feat = self.fc_edge(
                efeat).view(-1, self._num_heads, self._out_feats)

            el = (node_feat*self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (node_feat*self.attn_r).sum(dim=-1).unsqueeze(-1)
            ee = (edge_feat*self.attn_e).sum(dim=-1).unsqueeze(-1)
            graph.ndata['ft'] = node_feat
            graph.ndata['el'] = el
            graph.ndata['er'] = er
            graph.edata['ee'] = ee
            graph.apply_edges(fn.u_add_e('el', 'ee', 'el_prime'))
            graph.apply_edges(fn.e_add_v('el_prime', 'er', 'e'))
            e = self.leaky_relu(graph.edata['e'])
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.edata['efeat'] = edge_feat
            graph.update_all(self.msg_fn, fn.sum('m', 'ft'))
            rst = graph.ndata['ft']
            if self.residual:
                resval = self.res_fc(nfeat).view(
                    nfeat.shape[0], -1, self._out_feats)
                rst = rst + resval

            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class TemporalTransformerConv(nn.Module):
    def __init__(self,
                 edge_feats,
                 memory_feats,
                 temporal_encoder,
                 out_feats,
                 num_heads,
                 allow_zero_in_degree=False,
                 layers=1):
        '''Temporal Transformer model for TGN and TGAT

        Parameter
        ==========
        edge_feats : int
            number of edge features

        memory_feats : int
            dimension of memory vector

        temporal_encoder : torch.nn.Module
            compute fourier time encoding

        out_feats : int
            number of out features

        num_heads : int
            number of attention head

        allow_zero_in_degree : bool, optional
            If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
            since no message will be passed to those nodes. This is harmful for some applications
            causing silent performance regression. This module will raise a DGLError if it detects
            0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
            and let the users handle it by themselves. Defaults: ``False``.
        '''
        super(TemporalTransformerConv, self).__init__()
        self._edge_feats = edge_feats
        self._memory_feats = memory_feats
        self.temporal_encoder = temporal_encoder
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        self.layers = layers

        self.preprocessor = TemporalEdgePreprocess(self.temporal_encoder)
        self.edge_gatconv = EdgeGATConv(node_feats=self._memory_feats,
                                           edge_feats=self._edge_feats+self.temporal_encoder.dimension,
                                           out_feats=self._out_feats,
                                           num_heads=self._num_heads,
                                           feat_drop=0.6,
                                           attn_drop=0.6,
                                           residual=True,
                                           allow_zero_in_degree=allow_zero_in_degree)

        # self.layer_list = nn.ModuleList()
        # self.layer_list.append(EdgeGATConv(node_feats=self._memory_feats,
        #                                    edge_feats=self._edge_feats+self.temporal_encoder.dimension,
        #                                    out_feats=self._out_feats,
        #                                    num_heads=self._num_heads,
        #                                    feat_drop=0.6,
        #                                    attn_drop=0.6,
        #                                    residual=True,
        #                                    allow_zero_in_degree=allow_zero_in_degree))
        # for i in range(self.layers-1):
        #     self.layer_list.append(EdgeGATConv(node_feats=self._out_feats*self._num_heads,
        #                                        edge_feats=self._edge_feats+self.temporal_encoder.dimension,
        #                                        out_feats=self._out_feats,
        #                                        num_heads=self._num_heads,
        #                                        feat_drop=0.6,
        #                                        attn_drop=0.6,
        #                                        residual=True,
        #                                        allow_zero_in_degree=allow_zero_in_degree))

    def forward(self, graph, memory):
        graph = graph.local_var()
        # graph.ndata['timestamp'] = ts
        efeat = self.preprocessor(graph).float()
        rst = memory
        rst =  self.edge_gatconv(graph, rst, efeat).mean(1)
        # for i in range(self.layers-1):
        #     rst = self.layer_list[i](graph, rst, efeat).flatten(1)
        # rst = self.layer_list[-1](graph, rst, efeat).mean(1)
        return rst


def getModel(feature_dim, hidden_dim, num_nodes, device, gnn_param = None):
    if gnn_param is not None:
        # num_heads = 8, layers = 1
        gnn = TGNN(feature_dim, gnn_param['dim_out'], num_nodes, device, num_heads=gnn_param['att_head'], layers=gnn_param['layer']).to(device)
    else:
        gnn = TGNN(feature_dim, hidden_dim, num_nodes, device).to(device)
    # pred = MLPPredictor(hidden_dim).to(device)
    return {"gnn": gnn}

def getOptimizer(model, lr):
    return torch.optim.Adam(model['gnn'].parameters(),lr=lr,)