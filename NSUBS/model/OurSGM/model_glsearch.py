from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.our_conv import OurGATConv
from utils import OurTimer

import collections
import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from collections import defaultdict

from NSUBS.model.OurSGM.utils_nn import MLP

UBD_IDX = 0

class Interact_Encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, stride, heads):
        super(Interact_Encoder, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = int(dim_hidden/stride*heads)
        assert dim_hidden % stride == 0
        self.encoder = MLP(self.dim_in, dim_hidden)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=heads, kernel_size=stride, stride=stride)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=heads, kernel_size=stride, stride=stride)

    def forward(self, X1, X2):
        assert len(X1.shape) == len(X2.shape) == 2
        assert X1.shape[1] == X2.shape[1]
        Z1, Z2 = torch.sum(X1, dim=0), torch.sum(X1, dim=0)
        Z = \
            F.tanh(
                self.conv1(self.encoder(Z1).view(1,1,-1)) +
                self.conv1(self.encoder(Z2).view(1,1,-1))
            ).view(-1)
        return Z

class GLS(torch.nn.Module):
    def __init__(self):
        super(GLS, self).__init__()
        self.gnn_type = FLAGS.gnn_type
        self.interact_encoder_graph = Interact_Encoder(FLAGS.d_gnn, FLAGS.d_gnn, 16, 8)
        self.interact_encoder_subgraph = Interact_Encoder(FLAGS.d_gnn, FLAGS.d_gnn, 16, 8)
        self.interact_encoder_unconnected_bidomain = Interact_Encoder(FLAGS.d_gnn, FLAGS.d_gnn, 16, 8)
        self.interact_encoder_connected_bidomain = Interact_Encoder(FLAGS.d_gnn, FLAGS.d_gnn, 16, 8)
        # print(
        #     'MLP_final_in:',
        #     self.interact_encoder_graph.dim_out + self.interact_encoder_subgraph.dim_out +
        #     self.interact_encoder_unconnected_bidomain.dim_out + self.interact_encoder_connected_bidomain.dim_out
        # )
        # print(
        #     'interact_readout_in/out:',
        #     self.interact_encoder_connected_bidomain.dim_out
        # )
        self.MLP_readout = \
            MLP(
                self.interact_encoder_connected_bidomain.dim_out,
                self.interact_encoder_connected_bidomain.dim_out
            )
        self.MLP_final = \
            MLP(
                self.interact_encoder_graph.dim_out + self.interact_encoder_subgraph.dim_out +
                self.interact_encoder_unconnected_bidomain.dim_out + self.interact_encoder_connected_bidomain.dim_out,
                1, activation_type='elu', num_hidden_lyr=5
            )

    def print_statistics(self, CS_u2v, bdc_li_u_li_v_li):
        u_li_CS = [u for u in CS_u2v.keys()]
        v_set_li_CS = [set(v_li) for v_li in CS_u2v.values()]
        v_set_li_BD = [set(bdc[1]) for bdc in bdc_li_u_li_v_li]
        u_set_li_BD_u = [set(bdc[0]) for bdc in bdc_li_u_li_v_li]
        BDs_in_CS = \
            np.array(
                [
                    (
                        len([1 for v_set_BD in v_set_li_BD if len(v_set_BD.intersection(v_set_CS)) > 0]) + \
                        (1 if u not in set().union(*u_set_li_BD_u) else 0)
                    ) \
                    for (u, v_set_CS) in zip(u_li_CS, v_set_li_CS)
                ]
            )
        # for elt, u, v_li in zip(BDs_in_CS, u_li_CS, v_set_li_CS):
        #     if elt == 0:
        #         assert False
        CSs_in_BD = \
            np.array(
                [
                    len([1 for v_set_CS in v_set_li_CS if len(v_set_BD.intersection(v_set_CS)) > 0]) \
                    for v_set_BD in v_set_li_BD
                ]
            )
        CSs_in_BD_u = np.array([len(u_set) for u_set in u_set_li_BD_u])
        print('--------------------')
        print(f'BDs_in_CS: {np.mean(BDs_in_CS)}\t(std:{np.std(BDs_in_CS)}, len:{BDs_in_CS.shape})')
        print(f'CSs_in_BD: {np.mean(CSs_in_BD)}\t(std:{np.std(CSs_in_BD)}, len:{CSs_in_BD.shape})')
        print(f'CSs_in_BD_u: {np.mean(CSs_in_BD_u)}\t(std:{np.std(CSs_in_BD_u)}, len:{CSs_in_BD_u.shape})')
        print('--------------------')

    def forward(self, gq, gt, u, v_li, nn_map, CS_u2v,
                graph_filter=None, filter_key=None):
        u, v_li = int(u), [int(v) for v in v_li]
        out = []
        X1, X2 = gq.x_encoded, gt.x_encoded
        for v in v_li:
            bdc_li_u_li_v_li, bdu_u_li, bdu_v_li = self._execute_action(u, v, gq.nx_graph, gt.nx_graph, nn_map)
            # if len(bdc_li_u_li_v_li) > 0:
            #     self.print_statistics(CS_u2v, bdc_li_u_li_v_li)
            r_pred = self._dvn(X1, X2, nn_map, u, v, bdc_li_u_li_v_li, bdu_u_li, bdu_v_li)
            out.append(r_pred)
        out = torch.stack(out).view(-1)
        return out, X1, X2

    def _dvn(self, X1, X2, nn_map, u, v, bdc_li_u_li_v_li, bdu_u_li, bdu_v_li):
        HG = self.interact_encoder_graph(X1, X2)
        HS = self.interact_encoder_subgraph(X1[list(nn_map.keys())+[u]], X2[list(nn_map.keys())+[v]])
        HBDU = self.interact_encoder_unconnected_bidomain(X1[bdu_u_li], X2[bdu_v_li])
        HBDC = torch.zeros(self.interact_encoder_connected_bidomain.dim_out, dtype=torch.float, device=FLAGS.device)
        if len(bdc_li_u_li_v_li) > 0:
            HBDC = HBDC + \
                self.MLP_readout(
                    torch.sum(torch.stack(
                        [
                            self.interact_encoder_connected_bidomain(X1[u_li], X2[v_li]) \
                            for u_li, v_li in bdc_li_u_li_v_li
                        ], dim=0), dim=0
                    )
                )
        Q = self.MLP_final(torch.cat([HG, HS, HBDU, HBDC], dim=0))
        return Q

    def _execute_action(self, u, v, gq, gt, nn_map):
        # very slow implementation
        bdc2_u_li_v_li = {}
        candidate_us = set().union(*[set(gq.neighbors(u)) for u in nn_map.keys()]) - set(nn_map.keys())
        candidate_vs = set().union(*[set(gt.neighbors(v)) for v in nn_map.values()]) - set(nn_map.values())
        for u in candidate_us:
            bd_key = frozenset([nn_map[u_prime] for u_prime in gq.neighbors(u) if u_prime in nn_map])
            if bd_key in bdc2_u_li_v_li:
                bdc2_u_li_v_li[bd_key][0].append(u)
            else:
                bdc2_u_li_v_li[bd_key] = ([u], [])
        for v in candidate_vs:
            bd_key = frozenset(set(nn_map.values()).intersection(set(gt.neighbors(v))))
            if bd_key in bdc2_u_li_v_li:
                bdc2_u_li_v_li[bd_key][1].append(v)
        bdc_li_u_li_v_li = [bdc for bdc in bdc2_u_li_v_li.values() if len(bdc[1]) > 0]

        bdu_u_li = list(set(gq.nodes()) - candidate_us - set(nn_map.keys()) - {u})
        bdu_v_li = list(set(gt.nodes()) - candidate_vs - set(nn_map.values()) - {v})
        return bdc_li_u_li_v_li, bdu_u_li, bdu_v_li
