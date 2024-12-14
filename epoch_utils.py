import torch
import tqdm
from visualize import create_nx_multigraph, visualize_multigraph
import numpy as np

from dgl_utils import getGraph


# def tgb_mrr_eval(loader, model, neg_sampler, mode='val'):
#     # neg_samples = 1
#     model.eval()
#     # aps = list()
#     # aucs_mrrs = list()
#     # start_idx = 0
#     # if mode == 'val':
#     #     eval_df = val_dataloader#df[train_edge_end:val_edge_end]
#     #     start_idx = train_edge_end
#     # elif mode == 'test':
#     #     eval_df = test_dataloader#df[val_edge_end:]
#     #     neg_samples = args.eval_neg_samples
#     #     start_idx = val_edge_end
#     # elif mode == 'train':
#     #     eval_df = train_dataloader#df[:train_edge_end]
#     eval_df = loader
#     perf_list = []
#     with torch.no_grad():
#         total_loss = 0
#         # for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
#         for batch in eval_df:

#             neg_batch_list = neg_sampler.query_batch(batch['src'], batch['dst'], batch['t'], split_mode=mode)
#             # pos_src = batch['src'].numpy()
#             # pos_dst = batch['dst'].numpy()
#             # pos_ts = batch['t'].numpy()

#             for idx, neg_batch in enumerate(neg_batch_list):

#                 # src = np.full((1 + len(neg_batch),), pos_src[idx])
#                 # dst = np.concatenate(([[pos_dst[idx]], np.array(neg_batch)]),axis=0,)
#                 # root_nodes = np.concatenate([src, dst])
#                 ts = np.full((2+len(neg_batch),), pos_ts[idx])

#                 root_nodes = np.concatenate(([[pos_src[idx]],[pos_dst[idx]],np.array(neg_batch)]),axis=0,)
            
#                 # # root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
#                 # # ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
#                 # root_nodes =  np.concatenate([batch['src'].numpy(), batch['dst'].numpy(), neg_link_sampler.sample(batch['src'].shape[0])])
#                 # # ts =  np.concatenate([batch.t.numpy(), batch.t.numpy(), batch.t.numpy()])
#                 # ts = np.tile(batch['t'].numpy(), neg_samples + 2).astype(np.float32)
#             # 
#                 # if sampler is not None:
#                 #     if 'no_neg' in sample_param and sample_param['no_neg']:
#                 #         pos_root_end = len(rows) * 2
#                 #         sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
#                 #     else:
#                 sampler.sample(root_nodes, ts)
#                 ret = sampler.get_ret()
#                 if gnn_param['arch'] != 'identity':
#                     mfgs = to_dgl_blocks(ret, sample_param['history'])
#                 else:
#                     mfgs = node_to_dgl_blocks(root_nodes, ts)
#                 mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
#                 if mailbox is not None:
#                     mailbox.prep_input_mails(mfgs[0])
#                 pred_pos, pred_neg = model(mfgs, neg_samples=len(neg_batch))
#                 # print(pred_pos)
#                 # print(pred_neg[:20])


#                 # total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
#                 # total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
#                 y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
#                 # print(y_pred[:20])
#                 # input()
#                 input_dict = {
#                     "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
#                     "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
#                     "eval_metric": [metric],
#                 }
#                 perf_list.append(evaluator.eval(input_dict)[metric])
#     perf_metrics = float(torch.tensor(perf_list).mean())
#     return perf_metrics





@torch.no_grad()
def test(model, feats, loader, neighbor_loader, neg_sampler, assoc, device, optimizer, criterion, evaluator, metric, split_mode):

    # model['memory'].eval()
    model['gnn'].eval()
    # model['link_pred'].eval()

    perf_list = []

#     # n_id_obs = torch.empty(0, dtype=torch.long, device=device) 
#     # z_exp_obs = torch.zeros(1, MEM_DIM, device=device) 

    for batch in loader:
        src, pos_dst, t, msg, b = batch['src'], batch['dst'], batch['t'], batch['msg'], batch['b']
#         total += src.shape[0]
#         # pos_src, pos_dst, pos_t, pos_msg = (
#         #     pos_batch.src,
#         #     pos_batch.dst,
#         #     pos_batch.t,
#         #     pos_batch.msg,
#         # )

#         n_id_pos = torch.cat([src, pos_dst]).unique()
#         # new_nodes = n_id_pos[~torch.isin(n_id_pos, n_id_obs)] 
#         # n_id_seen = n_id_pos[~torch.isin(n_id_pos, new_nodes)] 
#         # n_id_obs = torch.cat((n_id_obs, new_nodes), dim=0).unique() 
        
        neg_batch_list = neg_sampler.query_batch(src, pos_dst, t, split_mode=split_mode)
        # print(len(neg_batch_list))
        
        # print(neg_batch_tensor.shape)
        # input("stopped by input 119")
        min_size =  99999999
        for idx, neg_batch in enumerate(neg_batch_list):
              if len(neg_batch) < min_size:
                  min_size = len(neg_batch)
            #   if len(neg_batch) != 999:
            #     print(len(neg_batch))
            #     input("stopped by input 122")
        neg_batch_list = [row[:min_size] for row in neg_batch_list]
        neg_dst = torch.tensor(neg_batch_list)
        # print(neg_dst.shape)
        # print(src.shape)
        
        k = b.max() + 1
        # print(k)
        msg = msg.to(device)
        t = t.to(device)
        # src = src.to(device)
        # pos_dst = pos_dst.to(device)
        # neg_dst = neg_dst.to(device)

        srcs = [src[b == i] for i in range(k)]
        pos_dsts = [pos_dst[b == i] for i in range(k)]
        neg_dsts = [neg_dst[b == i] for i in range(k)]
        tx = [t[b == i] for i in range(k)]
        msgs = [msg[b == i] for i in range(k)]
        # print("done with spliting")
        n_id = torch.cat([src, pos_dst, neg_dst.view(-1)]).unique().to(device)
        # print("n_id: ", n_id.shape)
        n_id, edge_index, e_id, batch_t = neighbor_loader(n_id)
        batch_feats = feats[e_id.cpu()].to(device)
        g = getGraph(edge_index[0], edge_index[1], n_id)
        pos_out, neg_out  = model['gnn'](g, batch_feats, batch_t, (srcs, pos_dsts, neg_dsts, tx, msgs,  neighbor_loader._assoc), neg_samples=neg_dst.shape[1])
        neg_out = neg_out.view(pos_out.shape[0], -1, 1)
        # print("pos_out shape: ", pos_out.shape)
        # print("neg_out_shape: ", neg_out.shape)
        # input_dict = {
        #         "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
        #         "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
        #         "eval_metric": [metric],
        #     }
        input_dict = {
                "y_pred_pos": np.array(pos_out.squeeze(dim=-1).cpu()),
                "y_pred_neg": np.array(neg_out.squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
        perf_list.append(evaluator.eval(input_dict)[metric])
        




        #       input()
#             src = torch.full((1 + len(neg_batch),), src[idx], device=device)
#             dst = torch.tensor(
#                 np.concatenate(
#                     ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
#                     axis=0,
#                 ),
#                 device=device,
#             )

#             n_id = torch.cat([src, dst]).unique()
#             # n_id_seen_neg = n_id_seen[torch.isin(n_id_seen, n_id)]
#             n_id, edge_index, e_id = neighbor_loader(n_id)
#             assoc[n_id] = torch.arange(n_id.size(0), device=device)

#             z, last_update = model['memory'](n_id)
#             z_exp = z_exp_obs[n_id_seen_neg].detach() 
#             z[assoc[n_id_seen_neg]] = z_exp 

#             z = model['gnn'](
#                 z,
#                 last_update,
#                 edge_index,
#                 data.t[e_id].to(device),
#                 data.msg[e_id].to(device),
#             )

#             y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

#             input_dict = {
#                 "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
#                 "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
#                 "eval_metric": [metric],
#             }
#             perf_list.append(evaluator.eval(input_dict)[metric])

#         model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        # neighbor_loader.insert(pos_src, pos_dst)
        neighbor_loader.insert(src.to(device), pos_dst.to(device), t.to(device))

#         x_obs = model['memory'].memory

#         z_exp_obs = exp_gnn(x_obs, cayley_g) 

    perf_metrics = float(torch.tensor(perf_list).mean())

    return perf_metrics


def train(model, feats, train_loader, neighbor_loader, neg_dest_sampler, assoc, device, optimizer, criterion):

    # model['memory'].train()
    # model['gnn'].train()
    # model['link_pred'].train()

    # model['memory'].reset_state()  
    neighbor_loader.reset_state()  

    n_id_obs = torch.empty(0, dtype=torch.long, device=device) 
    # z_exp_obs = torch.zeros(1, MEM_DIM, device=device) 

    total_loss = 0
    total = 0

    # sanity = 2
    # idx = 0
    for batch in train_loader:

        # #sanity
        # if idx > sanity:
        #     continue
        # idx += 1
        # batch = batch.to(device)
        # optimizer.zero_grad()

        src, pos_dst, t, msg, b = batch['src'], batch['dst'], batch['t'], batch['msg'], batch['b']
        total += src.shape[0]
        neg_dst = neg_dest_sampler.sample(pos_dst)


        k = b.max() + 1
        # print(k)
        msg = msg.to(device)
        t = t.to(device)
        # src = src.to(device)
        # pos_dst = pos_dst.to(device)
        # neg_dst = neg_dst.to(device)

        srcs = [src[b == i] for i in range(k)]
        pos_dsts = [pos_dst[b == i] for i in range(k)]
        neg_dsts = [neg_dst[b == i] for i in range(k)]
        tx = [t[b == i] for i in range(k)]
        msgs = [msg[b == i] for i in range(k)]

        n_id = torch.cat([src, pos_dst, neg_dst]).unique().to(device)
        # n_id = torch.cat([src, pos_dst, neg_dst]).to(device)
        # new_nodes = n_id[~torch.isin(n_id, n_id_obs)] 
        # n_id_seen = n_id[~torch.isin(n_id, new_nodes)] 
        # n_id_obs = torch.cat((n_id_obs, new_nodes), dim=0).unique() 
        n_id, edge_index, e_id, batch_t = neighbor_loader(n_id)

        # rev_nid = revmap(n_id)

        batch_feats = feats[e_id.cpu()].to(device)
        # print(batch_feats)



        # print("src: ", src)
        # print("pos_dst: ", pos_dst)
        # print("neg_dst: ", neg_dst)
        # print("n_id", n_id)
        # print("assoc: ", neighbor_loader._assoc[n_id])
        # print("e_id:", e_id)
        # print("batch_t: ", batch_t)
        # print("edge_index",edge_index)

        # visualize_multigraph(create_nx_multigraph(n_id,e_id, edge_index))


        g = getGraph(edge_index[0], edge_index[1], n_id)
        pos_out, neg_out  = model['gnn'](g, batch_feats, batch_t, (srcs, pos_dsts, neg_dsts, tx, msgs,  neighbor_loader._assoc))
        # print("pos score: ", pos_out)
        # print("neg score: ", neg_out)
        # print(pos_out.shape)
        # print(neg_out.shape)
        # has_inf = any(torch.isinf(param).any().item() for param in model['gnn'].parameters())

        # if has_inf:
        #     print("The model weights contain at least one infinite value.")





        # input("train epoch utils: ")
        
        # assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # z, last_update = model['memory'](n_id)
        # z_exp = z_exp_obs[n_id_seen].detach() 
        # z[assoc[n_id_seen]] = z_exp 
       
        # z = model['gnn'](
        #     z,
        #     last_update,
        #     edge_index,
        #     data.t[e_id].to(device),
        #     data.msg[e_id].to(device),
        # )

        # pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        # neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # model['memory'].update_state(src, pos_dst, t, msg)
        # print("112: ", torch.cuda.memory_allocated())
        neighbor_loader.insert(src.to(device), pos_dst.to(device), t.to(device))
        # print("114: ", torch.cuda.memory_allocated())

        loss.backward()
        optimizer.step()

        # x_obs = model['memory'].memory

        # z_exp_obs = exp_gnn(x_obs, cayley_g) 
        # model['memory'].detach()
        total_loss += float(loss) * src.shape[0]
        print("total_loss: ", total_loss, " total: ", total)
    
    return total_loss / total
    # return None
