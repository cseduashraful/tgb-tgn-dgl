import argparse
import os

from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


import torch
import time
import random
import dgl
import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

# from temporal_dataset import TemporalGraphDataset
from utils import parse_config, getData
from sampler_core import ParallelSampler


parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--use_inductive', action='store_true')
parser.add_argument('--rand_edge_features', type=int, default=0, help='use random edge featrues')
parser.add_argument('--rand_node_features', type=int, default=0, help='use random node featrues')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, train_dataset, val_dataset, test_dataset, neg_sampler, evaluator, metric = getData(args.data)
g, sample_param, memory_param, gnn_param, train_param = parse_config(args.config)

train_dataloader = DataLoader(train_dataset, batch_size= train_param['batch_size'], shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size= train_param['batch_size'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size= train_param['batch_size'], shuffle=False)

train_edge_end = train_dataset.src.shape[0]#df[df['ext_roll'].gt(0)].index[0]
val_edge_end = train_edge_end + val_dataset.src.shape[0] #df[df['ext_roll'].gt(1)].index[0]

gnn_dim_node = 0 #if node_feats is None else node_feats.shape[1]
gnn_dim_edge = dataset.edge_feat.shape[1]#0 if edge_feats is None else edge_feats.shape[1]
edge_feats  = dataset.edge_feat
node_feats = None

tgl_sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))