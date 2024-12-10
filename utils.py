import torch
import os
import yaml
# import dgl
import time
import pandas as pd
import numpy as np

from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from torch.utils.data import DataLoader

from temporal_dataset import TemporalGraphDataset
from dependencyGraph import dependecyAwareBatch as dab


def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def getDataWithDependecyBlock(DATA, train_param):
    dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()
    data = dataset.get_TemporalData()
    # data = data.to(device)
    metric = dataset.eval_metric

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    dataset.load_val_ns()
    dataset.load_test_ns()
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=DATA)

    train_dataset = TemporalGraphDataset(train_data.src, train_data.dst, train_data.t, train_data.msg)
    val_dataset = TemporalGraphDataset(val_data.src, val_data.dst, val_data.t, val_data.msg)
    test_dataset = TemporalGraphDataset(test_data.src, test_data.dst, test_data.t, test_data.msg)

    train_dataloader = DataLoader(train_dataset, batch_size= train_param['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size= train_param['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size= train_param['batch_size'], shuffle=False)
    train_blocks = dab(train_dataloader)
    test_blocks = dab(val_dataloader)
    val_blocks = dab(test_dataloader)
    train_dataset = TemporalGraphDataset(train_data.src, train_data.dst, train_data.t, train_data.msg, batch=train_blocks)
    val_dataset = TemporalGraphDataset(val_data.src, val_data.dst, val_data.t, val_data.msg, batch=val_blocks)
    test_dataset = TemporalGraphDataset(test_data.src, test_data.dst, test_data.t, test_data.msg, batch=test_blocks)

    train_dataloader = DataLoader(train_dataset, batch_size= train_param['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size= train_param['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size= train_param['batch_size'], shuffle=False)
    return data, train_dataloader, val_dataloader, test_dataloader, neg_sampler, evaluator, metric



def getData(DATA, batches = {'train': None, 'val': None, 'test':None}):
    dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
    g = np.load('DATA/{}/ext_full.npz'.format(DATA))
    
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()
    # data = data.to(device)
    metric = dataset.eval_metric

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    dataset.load_val_ns()
    dataset.load_test_ns()
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=DATA)

    

    train_dataset = TemporalGraphDataset(train_data.src, train_data.dst, train_data.t, train_data.msg, batch=batches['train'])
    val_dataset = TemporalGraphDataset(val_data.src, val_data.dst, val_data.t, val_data.msg, batch=batches['val'])
    test_dataset = TemporalGraphDataset(test_data.src, test_data.dst, test_data.t, test_data.msg, batch=batches['test'])

    return g, dataset, train_dataset, val_dataset, test_dataset, neg_sampler, evaluator, metric
