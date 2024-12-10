import argparse
import yaml
import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import dgl
from sampler_core import ParallelSampler, TemporalGraphBlock

class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

class NegLinkSamplerDest:

    def __init__(self, dst_nodes):
        self.dst_nodes = dst_nodes.tolist()

    def sample(self, pos_dst):
        negdst = np.random.choice(self.dst_nodes, size=len(pos_dst))
        same_value_indices = np.where(negdst == pos_dst)
        # print(same_value_indices)
        # print(pos_dst[same_value_indices])
        # print(negdst[same_value_indices])
        # input("negsampler")
        if len(same_value_indices[0])>0:
            newdst = self.sample(pos_dst[same_value_indices])
            negdst[same_value_indices] = newdst
        return negdst
    

class NegLinkInductiveSampler:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, n):
        return np.random.choice(self.nodes, size=n)
    
# if __name__ == '__main__':
#     parser=argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, help='dataset name')
#     parser.add_argument('--config', type=str, help='path to config file')
#     parser.add_argument('--batch_size', type=int, default=600, help='path to config file')
#     parser.add_argument('--num_thread', type=int, default=64, help='number of thread')
#     args=parser.parse_args()

#     df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
#     g = np.load('DATA/{}/ext_full.npz'.format(args.data))
#     sample_config = yaml.safe_load(open(args.config, 'r'))['sampling'][0]

#     sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
#                               args.num_thread, 1, sample_config['layer'], sample_config['neighbor'],
#                               sample_config['strategy']=='recent', sample_config['prop_time'],
#                               sample_config['history'], float(sample_config['duration']))

#     num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
#     neg_link_sampler = NegLinkSampler(num_nodes)

#     tot_time = 0
#     ptr_time = 0
#     coo_time = 0
#     sea_time = 0
#     sam_time = 0
#     uni_time = 0
#     total_nodes = 0
#     unique_nodes = 0
#     for _, rows in tqdm(df.groupby(df.index // args.batch_size), total=len(df) // args.batch_size):
#         root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
#         ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
#         sampler.sample(root_nodes, ts)
#         ret = sampler.get_ret()
#         tot_time += ret[0].tot_time()
#         ptr_time += ret[0].ptr_time()
#         coo_time += ret[0].coo_time()
#         sea_time += ret[0].search_time()
#         sam_time += ret[0].sample_time()
#         # for i in range(sample_config['history']):
#         #     total_nodes += ret[i].dim_in() - ret[i].dim_out()
#         #     unique_nodes += ret[i].dim_in() - ret[i].dim_out()
#         #     if ret[i].dim_in() > ret[i].dim_out():
#         #         ts = torch.from_numpy(ret[i].ts()[ret[i].dim_out():])
#         #         nid = torch.from_numpy(ret[i].nodes()[ret[i].dim_out():]).float()
#         #         nts = torch.stack([ts,nid],dim=1).cuda()
#         #         uni_t_s = time.time()
#         #         unts, idx = torch.unique(nts, dim=0, return_inverse=True)
#         #         uni_time += time.time() - uni_t_s
#         #         total_nodes += idx.shape[0]
#         #         unique_nodes += unts.shape[0]

#     print('total time  : {:.4f}'.format(tot_time))
#     print('pointer time: {:.4f}'.format(ptr_time))
#     print('coo time    : {:.4f}'.format(coo_time))
#     print('search time : {:.4f}'.format(sea_time))
#     print('sample time : {:.4f}'.format(sam_time))
#     # print('unique time : {:.4f}'.format(uni_time))
#     # print('unique per  : {:.4f}'.format(unique_nodes / total_nodes))


    


def to_dgl_graph(ret, hist, edge_feats, root_nodes, reverse=False, cuda=True, id_bak = False):
    mfgs =  list()
    id_name = dgl.NID
    if id_bak:
        id_name = "ID"
    for r in ret:
        # if use_sk and sk:
        #     print(r.col())
        #     print(r.row())
        #     print(r.nodes())

        #     k = input("utils ln 49: type stop to avoid blocking next time")
        #     if k  == 'stop':
        #         sk = False
        if not reverse:
            g = dgl.graph((r.col(), r.row()), num_nodes=len(r.nodes()))
        else:
            g = dgl.graph((r.row(), r.col()), num_nodes=len(r.nodes()))
        g.ndata[id_name] = torch.from_numpy(r.nodes())
        g.ndata['timestamp'] = torch.from_numpy(r.ts())
        # print("r.eid(): ", r.eid())
        ID = torch.from_numpy(r.eid())
        dt = torch.from_numpy(r.dts())[r.dim_out():]
        if edge_feats is not None:
            feats =  edge_feats[ID]
        # feat = embedding[feat_id]
        # print(root_nodes)
        self_loop =  False
        if self_loop:
    
            g.add_edges(torch.arange(0,len(root_nodes)), torch.arange(0,len(root_nodes)))
            
            g.edata[id_name] = torch.arange(0, g.number_of_edges(), dtype=torch.int64)


            # print(g.number_of_edges())
            # print(feats.shape)
            
            
            # print(zeros.shape)
            if edge_feats is not None:
                zeros  = torch.zeros(len(root_nodes), feats.shape[1])
                g.edata['feats'] = torch.cat((feats, zeros), dim=0)


            g.edata['ID'] = torch.cat((ID, torch.zeros(len(root_nodes))), dim=0)
            g.edata['timestamp'] = torch.cat((dt, torch.zeros(len(root_nodes))), dim=0)
        else:
            g.edata['ID'] = ID#torch.cat((ID, torch.zeros(len(root_nodes))), dim=0)
            g.edata['timestamp'] = dt#torch.cat((dt, torch.zeros(len(root_nodes))), dim=0)
            if edge_feats is not None:
                g.edata['feats'] = feats#torch.cat((feats, zeros), dim=0)
        

        if cuda:
            mfgs.append(g.to('cuda:0'))
        else:
            mfgs.append(g)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs
