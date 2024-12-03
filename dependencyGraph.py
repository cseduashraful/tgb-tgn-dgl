# from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import torch
# from tgb.utils.utils import get_args
# from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm 


def get_block(tss, src_b, dst_b):
    last_accessed_dict = {}
    access_dict = {}
    t_access_dict = {}
    block_ids = []
    for i, ts in enumerate(tss):
        edge_nodes  = torch.tensor([src_b[i], dst_b[i]])
        last_accessed_times = [last_accessed_dict[k.item()] if k.item() in last_accessed_dict.keys() else -1 for k in edge_nodes]
        block_id = max(last_accessed_times)+1 if len(last_accessed_times)>0 else 0
        for k in edge_nodes:
            last_accessed_dict[k.item()] =  block_id
            kitem = k.item()
            if k.item() in access_dict.keys():
                access_dict[kitem].append(block_id)
                t_access_dict[k.item()].append(ts)
            else:
                access_dict[kitem] = [block_id]
                t_access_dict[k.item()] = [ts]    
        block_ids.append(block_id)
    # print(".... ", max(block_ids))
    return block_ids
        



def dependecyAwareBatch(loader, flat = True):
    block_ids = []
    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t, _ = (
            pos_batch['src'],
            pos_batch['dst'],
            pos_batch['t'],
            pos_batch['msg'],
        )
        # print(pos_src)
        # print(pos_dst)
        # input("next")
        if flat:
            block_ids.extend(get_block(pos_t, pos_src, pos_dst))
        else:
            block_ids.append(get_block(pos_t, pos_src, pos_dst))
    return block_ids


















# args, _ = get_args()
# print("INFO: Arguments:", args)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DATA = "tgbl-wiki"
# BATCH_SIZE = args.bs

# dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
# train_mask = dataset.train_mask
# val_mask = dataset.val_mask
# test_mask = dataset.test_mask
# data = dataset.get_TemporalData()
# data = data.to(device)
# metric = dataset.eval_metric

# train_data = data[train_mask]
# val_data = data[val_mask]
# test_data = data[test_mask]

# # print(train_data)

# train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
# val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
# test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
# neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)


# train_blocks = dependecyAwareBatch(train_loader)
# test_blocks = dependecyAwareBatch(test_loader)
# val_blocks = dependecyAwareBatch(val_loader)

