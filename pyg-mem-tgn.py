import argparse
import os

from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


import torch
import time
import random
# import dgl
# import numpy as np

# from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

# from temporal_dataset import TemporalGraphDataset
from utils import parse_config, getDataWithDependecyBlock
# from sampler_core import ParallelSampler
from dependencyGraph import dependecyAwareBatch as dab
# from sampler import *
from neighbor_loader import LastNeighborLoader
from epoch_utils import train
from neg_sampler import NegLinkSamplerDest


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
# parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
# parser.add_argument('--model_name', type=str, default='', help='name of stored model')
# parser.add_argument('--use_inductive', action='store_true')
# parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
# parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
# parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

data, train_dataloader, val_dataloader, test_dataloader, neg_sampler, evaluator, metric = getDataWithDependecyBlock(args.data, train_param)
unique_destination_nodes =  torch.unique(data.dst)
# train_dataloader = DataLoader(train_dataset, batch_size= train_param['batch_size'], shuffle=False)
# val_dataloader = DataLoader(val_dataset, batch_size= train_param['batch_size'], shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size= train_param['batch_size'], shuffle=False)
# train_edge_end = train_dataset.src.shape[0]#df[df['ext_roll'].gt(0)].index[0]
# val_edge_end = train_edge_end + val_dataset.src.shape[0] #df[df['ext_roll'].gt(1)].index[0]

# gnn_dim_node = 0 #if node_feats is None else node_feats.shape[1]
# gnn_dim_edge = dataset.edge_feat.shape[1]#0 if edge_feats is None else edge_feats.shape[1]
# edge_feats  = dataset.edge_feat
# node_feats = None

print("Neighbor::: ", sample_param['neighbor'][0])
neg_dest_sampler = NegLinkSamplerDest(unique_destination_nodes)
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=sample_param['neighbor'][0], device=device)

start_time = time.time()
for e in range(train_param['epoch']):
    print('Epoch {:d}:'.format(e))
    loss = train(train_dataloader, neighbor_loader, neg_dest_sampler, assoc, device)
    # for batch in train_dataloader:
        
    #     # print("batch['src']: ", batch['src'])
    #     # print("batch['dst']: ", batch['dst'])
    #     # print("batch['b']: ", batch['b'])
    #     # print("batch['t']: ", batch['t'])
    #     # x = batch['src']
    #     loss = train()


end_time = time.time()
execution_time = end_time - start_time 
print(f"Execution Time: {execution_time:.6f} seconds")