import torch
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.memory_module import TGNMemory
# from modules.early_stopping import  EarlyStopMonitor


def getModel(feature_dim, hidden_dim, num_nodes, device):
    # gnn = TGNN(feature_dim, hidden_dim, num_nodes, device).to(device)
    # pred = MLPPredictor(hidden_dim).to(device)

    memory = TGNMemory(
        num_nodes,
        feature_dim, #data.msg.size(-1),
        hidden_dim, #MEM_DIM,
        hidden_dim, #TIME_DIM,
        message_module=IdentityMessage(feature_dim, hidden_dim, hidden_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=hidden_dim, #MEM_DIM,
        out_channels=hidden_dim, #EMB_DIM,
        msg_dim=feature_dim, #data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    # link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)
    link_pred = LinkPredictor(in_channels=hidden_dim).to(device)

    model = {'memory': memory,
            'gnn': gnn,
            'link_pred': link_pred}
    return model

def getOptimizer(model, lr):
    optimizer = torch.optim.Adam(
        set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
        lr=lr,
    )
    return optimizer