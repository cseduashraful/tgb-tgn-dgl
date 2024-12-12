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


#change based on dgl/pyg
from model_utils import getModel


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
print("feature dim: ", data.msg.shape[1])

print("Neighbor::: ", sample_param['neighbor'][0])
neg_dest_sampler = NegLinkSamplerDest(unique_destination_nodes)
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=sample_param['neighbor'][0], device=device)


model = getModel(data.msg.shape[1], gnn_param['dim_out'], device)
# optimizer = torch.optim.Adam(
#     set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
#     lr=LR,
# )
optimizer = torch.optim.Adam(model['gnn'].parameters(),lr=train_param['lr'],)
criterion = torch.nn.BCEWithLogitsLoss()

start_time = time.time()
for e in range(train_param['epoch']):
    # print('Epoch {:d}:'.format(e))
    trs = time.time()
    loss = train(model, data.msg, train_dataloader, neighbor_loader, neg_dest_sampler, assoc, device, optimizer, criterion)
    tre = time.time()
    print(f"Epoch: {e:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {tre-trs: .4f}")
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