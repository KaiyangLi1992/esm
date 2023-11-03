import torch
import torch.nn as nn

from NSUBS.model.OurSGM.config import FLAGS
from graphgps.network.gps_model import FeatureEncoder
from torch_geometric.graphgym.config import cfg

def create_preencoder(d_in_raw, d_in):
    if FLAGS.dvn_config['preencoder']['type'] == 'concat+mlp':
        preencoder = PreEncoderConcatSelectedOneHotAndMLP(d_in_raw, d_in)
    elif FLAGS.dvn_config['preencoder']['type'] == 'mlp':
        preencoder = PreEncoderMLP(d_in_raw, d_in)
    else:
        assert False
    return preencoder

def get_one_hot_labelling(N, sel_nodes):
    sel_nodes_one_hot = torch.zeros(N, dtype=torch.float32, device=FLAGS.device)
    sel_nodes_one_hot[sel_nodes] = 1
    sel_nodes_one_hot = torch.stack((sel_nodes_one_hot, 1-sel_nodes_one_hot), dim=-1)
    return sel_nodes_one_hot

class PreEncoderConcatSelectedOneHotAndMLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(PreEncoderConcatSelectedOneHotAndMLP, self).__init__()
        if dim_in + 2 != dim_out:
            # self.mlp_q = nn.Linear(dim_in + 2, dim_out-64)
            # self.mlp_t = nn.Linear(dim_in + 2, dim_out-64)
            self.mlp_map_q = nn.Linear(2, 8)
            self.mlp_map_t = nn.Linear(2, 8)
            self.feature_encoder_q = FeatureEncoder(1)
            self.feature_encoder_t = FeatureEncoder(1)
        else:
            self.mlp_q = self.mlp_t = lambda x: x

    def forward(self, Xq, Xt, nn_map,pyg_data_q,pyg_data_t):
        selected_one_hot_q = get_one_hot_labelling(Xq.shape[0], list(nn_map.keys()))
        selected_one_hot_t = get_one_hot_labelling(Xt.shape[0], list(nn_map.values()))
        selected_one_hot_q = self.mlp_map_q(selected_one_hot_q)
        selected_one_hot_t = self.mlp_map_t(selected_one_hot_t)

        Xq =  self.feature_encoder_q(pyg_data_q).x
        Xt =  self.feature_encoder_t(pyg_data_t).x
        # Xq = torch.cat((Xq, selected_one_hot_q), dim=1)
        # Xt = torch.cat((Xt, selected_one_hot_t), dim=1)
        # Xq = self.mlp_q(Xq)
        # Xt = self.mlp_t(Xt)
        # RWSEq = self.mlp_RWSEq(RWSEq)
        # RWSEt = self.mlp_RWSEt(RWSEt)
        Xq = torch.cat((Xq, selected_one_hot_q), dim=1)
        Xt = torch.cat((Xt, selected_one_hot_t), dim=1)
        return Xq, Xt

class PreEncoderMLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(PreEncoderMLP, self).__init__()
        self.mlp_q = nn.Linear(dim_in, dim_out)
        self.mlp_t = nn.Linear(dim_in, dim_out)

    def forward(self, Xq, Xt, *args):
        Xq = self.mlp_q(Xq)
        Xt = self.mlp_t(Xt)
        return Xq, Xt