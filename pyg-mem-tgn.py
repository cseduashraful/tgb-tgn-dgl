import argparse
import os
import torch
import time

# from tgb.utils.utils import get_args, set_random_seed, save_results
# from tgb.linkproppred.evaluate import Evaluator
# from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset



# import random
# from torch.utils.data import DataLoader

# from temporal_dataset import TemporalGraphDataset
from utils import parse_config, getDataWithDependecyBlock
from dependencyGraph import dependecyAwareBatch as dab
from neighbor_loader import LastNeighborLoader
from epoch_utils import train, test
from neg_sampler import NegLinkSamplerDest


#change based on dgl/pyg
from model_utils import getModel, getOptimizer
# from pyg_model_utils import getModel, getOptimizer


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
args=parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

data, train_dataloader, val_dataloader, test_dataloader, neg_sampler, evaluator, metric = getDataWithDependecyBlock(args.data, train_param)
unique_destination_nodes =  torch.unique(data.dst)

# print("feature dim: ", data.msg.shape[1])
# print("Neighbor::: ", sample_param['neighbor'][0])

neg_dest_sampler = NegLinkSamplerDest(unique_destination_nodes)
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=sample_param['neighbor'][0], device=device)


model = getModel(data.msg.shape[1], gnn_param['dim_out'], data.num_nodes, device, gnn_param=gnn_param)
optimizer = getOptimizer(model,train_param['lr'] )
criterion = torch.nn.BCEWithLogitsLoss()

start_time = time.time()
for e in range(train_param['epoch']):
    print('Epoch {:d}:'.format(e))
    trs = time.time()
    loss = train(model, data.msg, train_dataloader, neighbor_loader, neg_dest_sampler, assoc, device, optimizer, criterion)
    tre = time.time()
    print(f"Epoch: {e+1:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {tre-trs: .4f}")
    trs = time.time()
    perf_metric_val = test(model, data.msg, val_dataloader, neighbor_loader, neg_sampler, assoc, device, optimizer, criterion, evaluator, metric, 'val')
    tre = time.time()
    print(f"Validation {metric}: {perf_metric_val: .4f}, elapsed Time (s): {tre-trs: .4f}")
    
end_time = time.time()
execution_time = end_time - start_time 
print(f"Execution Time: {execution_time:.6f} seconds")