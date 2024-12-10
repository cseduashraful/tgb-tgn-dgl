import torch
import tqdm
def train(train_loader, neighbor_loader, neg_dest_sampler, assoc, device):

    # model['memory'].train()
    # model['gnn'].train()
    # model['link_pred'].train()

    # model['memory'].reset_state()  
    neighbor_loader.reset_state()  

    n_id_obs = torch.empty(0, dtype=torch.long, device=device) 
    # z_exp_obs = torch.zeros(1, MEM_DIM, device=device) 

    total_loss = 0
    for batch in train_loader:
        # batch = batch.to(device)
        # optimizer.zero_grad()

        src, pos_dst, t, msg, b = batch['src'], batch['dst'], batch['t'], batch['msg'], batch['b']
        neg_dst = neg_dest_sampler.sample(pos_dst)

        
        # neg_dst = torch.randint(
        #     min_dst_idx,
        #     max_dst_idx + 1,
        #     (src.size(0),),
        #     dtype=torch.long,
        #     device=device,
        # )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique().to(device)
        new_nodes = n_id[~torch.isin(n_id, n_id_obs)] 
        n_id_seen = n_id[~torch.isin(n_id, new_nodes)] 
        n_id_obs = torch.cat((n_id_obs, new_nodes), dim=0).unique() 
        n_id, edge_index, e_id = neighbor_loader(n_id)
        
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

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

        # loss = criterion(pos_out, torch.ones_like(pos_out))
        # loss += criterion(neg_out, torch.zeros_like(neg_out))

        # model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src.to(device), pos_dst.to(device))

        # loss.backward()
        # optimizer.step()

        # x_obs = model['memory'].memory

        # z_exp_obs = exp_gnn(x_obs, cayley_g) 
        # model['memory'].detach()
        # total_loss += float(loss) * batch.num_events
    
    # return total_loss / train_data.num_events
