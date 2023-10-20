from config import FLAGS
from our_conv import OurGATConv
from saver import saver
from utils import OurTimer

import collections
import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, GraphNorm
from collections import defaultdict

def create_gnn(gnn_type, n_layers, d_in, d_gnn, heads):
    if gnn_type == 'GCN':
        gnn_class = GCNConv
    elif gnn_type == 'GAT':
        gnn_class = GATConv
    elif gnn_type == 'GIN':
        gnn_class = GINConv
    elif gnn_type == 'OurGAT':
        gnn_class = OurGATConv
    else:
        raise ValueError(f'Unknown gnn_type {gnn_type}')
    if d_gnn % heads != 0:
        raise ValueError(f'd_gnn must be divisible by # of heads')
    if n_layers == 1:
        gnn = nn.ModuleList(
            [
                create_one_gnn(gnn_class, gnn_type, d_in, d_gnn // heads, heads)
            ]
        )
    else:
        gnn = nn.ModuleList(
            [create_one_gnn(gnn_class, gnn_type, d_in, d_gnn // heads, heads)] +
            [
                create_one_gnn(gnn_class, gnn_type, d_gnn, d_gnn // heads, heads)
                for _ in range(n_layers - 1)
            ]
        )
    return gnn

def create_one_gnn(gnn_class, gnn_type, d_in, d_out, heads=None):
    if gnn_type in ['GCN']:
        return gnn_class(d_in, d_out)
    elif gnn_type in ['GIN']:
        return gnn_class(nn.Sequential(nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ELU(),
                       nn.Linear(d_out, d_out)))
    elif gnn_type in ['GAT', 'OurGAT']:
        assert heads is not None
        return gnn_class(d_in, d_out, heads=heads)
    else:
        raise ValueError(f'Unknown gnn_type {FLAGS.gnn_type}')

class Encoder_MLP(torch.nn.Module):
    def __init__(self, d_in_raw):
        super(Encoder_MLP, self).__init__()
        self.mlp = nn.Linear(d_in_raw, FLAGS.d_enc)

    def __call__(self, gq, gt, CS_u2v):
        gq.x_encoded = self.mlp(gq.x)
        gt.x_encoded = self.mlp(gt.x)

class Encoder_GLSearch(torch.nn.Module):
    def __init__(self, d_in_raw):
        super(Encoder_GLSearch, self).__init__()
        self.n_layers = 3
        self.mlp = nn.Linear(d_in_raw, FLAGS.d_gnn)
        self.gnn = create_gnn('GAT', self.n_layers, FLAGS.d_gnn, FLAGS.d_gnn, 1)

    def __call__(self, gq, gt, CS_u2v):
        X_q, X_t = self.mlp(gq.x), self.mlp(gt.x)
        # X_q, X_t = gq.x, gt.x
        for i in range(self.n_layers):
            X_q = F.elu(self.gnn[i](X_q, gq.edge_index))
            X_t = F.elu(self.gnn[i](X_t, gq.edge_index))
        gq.x_encoded, gt.x_encoded = X_q, X_t

class Encoder_GCN(torch.nn.Module):
    def __init__(self, d_in_raw):
        super(Encoder_GCN, self).__init__()
        self.n_layers = 2
        self.mlp = nn.Linear(d_in_raw, FLAGS.d_enc)
        self.gnn = create_gnn('GCN', self.n_layers, FLAGS.d_enc, FLAGS.d_enc, 1)
        self.gn = GraphNorm(FLAGS.d_enc)

    def __call__(self, gq, gt, CS_u2v):
        X_q, X_t = gq.x, gt.x

        # run GCN
        X_q = self.mlp(X_q)
        for i in range(self.n_layers):
            # X_q = F.elu(self.gn(self.gnn[i](X_q, gq.edge_index)))
            X_q = F.elu(self.gnn[i](X_q, gq.edge_index))
            # X_q = X_q_new
        # consensus
        # X_q = X_q)
        X_t = torch.zeros((X_t.size(0), X_q.size(1)), device=FLAGS.device)
        X_norm = torch.ones((X_t.size(0), 1), device=FLAGS.device)
        for u, v_li in CS_u2v.items():
            X_t[v_li] += X_q[u]
            X_norm[v_li] += 1
        gq.x_encoded = X_q
        gt.x_encoded = X_t / X_norm
        # gq.x_encoded = F.normalize(X_q, p=2, dim=0)
        # gt.x_encoded = F.normalize(X_t, p=2, dim=0)
        # gq.x_encoded = F.normalize(F.normalize(X_q, p=2, dim=0), p=2, dim=1)
        # gt.x_encoded = F.normalize(F.normalize(X_t, p=2, dim=0), p=2, dim=1)

    def compute_edge_index_of_subgraph(self, g, sel_nodes):
        sel_g = nx.subgraph(g, sel_nodes)
        sel_g_edges = [[int(x) for x in lst] for lst in sel_g.edges()]
        edge_index = torch.LongTensor(sel_g_edges).t().contiguous().to(FLAGS.device)
        return edge_index

class DGMC(torch.nn.Module):
    def __init__(self):
        super(DGMC, self).__init__()
        self.bilinear = nn.ModuleList(
            DynamicBilinear(FLAGS.d_gnn, FLAGS.d_bilinear)
            for _ in range(FLAGS.n_layers)
        )
        d_mlp = FLAGS.d_bilinear * FLAGS.n_layers
        self.mlp = nn.Sequential(
            nn.Linear(d_mlp, d_mlp // 2),
            nn.ReLU(),
            nn.Linear(d_mlp // 2, d_mlp // 4),
            nn.ReLU(),
            nn.Linear(d_mlp // 4, 1)
        )
        # self._create_gnn()
        self.gnn = create_gnn(FLAGS.gnn_type, FLAGS.n_layers, FLAGS.d_enc, FLAGS.d_gnn, FLAGS.heads)
        self.gnn_type = FLAGS.gnn_type
        self.n_layers = FLAGS.n_layers
        self.gnn_out_q = []  # TODO: check: this is unused

    def _call_gnn(self, i, X_q, edge_index_q, X_t, edge_index_t, cached_tensors):
        if self.gnn_type in ['GAT', 'GCN']:
            if cached_tensors is None:
                X_q = self.gnn[i](X_q, edge_index_q)
            else:
                X_q = cached_tensors['X_q'][i]
            if X_t is not None:
                # print('X_t.shape', X_t.shape)
                # print('edge_index_t.shape', edge_index_t.shape)
                X_t = self.gnn[i](X_t, edge_index_t)
                # try:
                #     X_t = self.gnn[i](X_t, edge_index_t)
                # except IndexError as e:
                #     saver.log_info(e)
                #     raise RuntimeError()
        elif self.gnn_type == 'OurGAT':
            if cached_tensors is None:
                X_q = self.gnn[i].forward_gq(X_q, edge_index_q)
            else:
                X_q = cached_tensors['X_q'][i]
            if X_t is not None:
                X_t = self.gnn[i].forward_gt(X_t, edge_index_t, X_q)
            # if cached_tensors is None:
            #     X_q, X_t = self.gnn[i](X_q, edge_index_q, X_t, edge_index_t, None)
            # else:
            #     X_q = cached_tensors['X_q'][i]
            #     X_q, X_t = self.gnn[i](None, edge_index_q, X_t, edge_index_t, X_q)
        else:
            assert False
        return X_q, X_t

    def cache_gq(self, X_q, edge_index_q):
        cached_tensors = defaultdict(dict)
        for i in range(self.n_layers):
            X_q, _ = self._call_gnn(i, X_q, edge_index_q, None, None, None)
            if i == self.n_layers - 1:
                X_q = F.normalize(X_q, p=2, dim=1)
            else:
                X_q = F.elu(F.normalize(X_q, p=2, dim=1))
            cached_tensors['X_q'][i] = X_q
        return cached_tensors
        # pass

    # def _get_X_edge_index_tensors(self, gq, gt):
    #     X_q, edge_index_q, X_t = gq.x_encoded, gq.edge_index, gt.x_encoded
    #     # if graph_filter is not None and filter_key is not None and FLAGS.is_efficient_gt:
    #     #     # print('@@@@')
    #     #     edge_index_t = graph_filter.filter_graph(gq, gt, filter_key)
    #     # else:
    #     #     # print('$$$')
    #     edge_index_t = gt.edge_index
    #     return X_q, edge_index_q, X_t, edge_index_t

    def forward(self, gq, gt, u, v_li, nn_map, CS_u2v):
        X_q, edge_index_q, X_t, edge_index_t = gq.x_encoded, gq.edge_index, gt.x_encoded, gt.edge_index
        cached_tensors = None
        # self.our_timer = OurTimer()
        # print('###', u, v_li)
        u, v_li = int(u), [int(v) for v in v_li]
        out = []

        # print('initial ')
        # print('u embs:',X_q[u])
        # print('v_li:', v_li)
        # print('v embs:', X_t[v_li])
        # print('X_t', X_t)
        # print('edge_index_t', edge_index_t)
        # print('cached_tensors', cached_tensors)
        # assert len(v_li) == len(set(v_li)), f'v_li duplicates {v_li}'
        # exit()

        # self.our_timer.time_and_clear('setup')
        for i in range(self.n_layers):
            # X_q = self.gnn_out_q[i]
            X_q, X_t = self._call_gnn(
                i, X_q, edge_index_q, X_t, edge_index_t, cached_tensors)
            # self.our_timer.time_and_clear(f'layer{i} call_gnn done')

            # print('X_q', X_q)
            # print('X_t @@@', X_t)
            # exit()

            if i == self.n_layers - 1:
                X_q = F.normalize(X_q, p=2, dim=1)
                X_t = F.normalize(X_t, p=2,
                                  dim=1)  # TODO: not scalable when gt is too large? cache X_t?
            else:
                X_q = F.elu(F.normalize(X_q, p=2, dim=1))
                X_t = F.elu(F.normalize(X_t, p=2, dim=1))
                # self.our_timer.time_and_clear(f'layer{i} normalize_gnn done')
            # else:
            #     assert self.gnn_type == 'GAT' and not self.training
            #     X_q = cached_tensors['X_q'][i]
            #     X_t = self.gnn[i](X_t, edge_index_t)
            # self.our_timer.time_and_clear(f'layer{i} gnn part done')
            if FLAGS.use_consensus:
                if FLAGS.is_efficient_consensus:
                    if len(nn_map) > 0:
                        u_consensus_li, v_consensus_li = np.array(
                            [[int(u), int(v)] for (u, v) in nn_map.items()]).T
                        X_t[v_consensus_li] = X_q[u_consensus_li]
                else:
                    for u_consensus, v_li_consensus in CS_u2v.items():
                        u_consensus, v_li_consensus = int(u_consensus), [int(vc) for vc in
                                                                         v_li_consensus]
                        if u_consensus in nn_map:
                            X_t[nn_map[u_consensus]] = X_q[u_consensus]
                        else:
                            att = F.softmax(torch.matmul(X_q[u_consensus].view(1, -1),
                                                         X_t[v_li_consensus].transpose(0, 1)), dim=-1)
                            X_t[v_li_consensus] = att.view(-1, 1) * X_q[u_consensus].view(1, -1) + (
                                        1 - att).view(-1, 1) * X_t[v_li_consensus]
            # self.our_timer.time_and_clear(f'layer{i} consensus')
            # if i == 1:
            #     print('$$$$$$$$$$')
            #     # print('u embs:',X_q[u])
            #     print('v_li:', v_li)
            #     print('v embs:',X_t[v_li])
            #     print('X_t', X_t)
            #     assert len(v_li) == len(set(v_li)), f'v_li duplicates {v_li}'

            out.append(
                self.bilinear[i](X_q[u], X_t[v_li]).view(-1, len(v_li)).transpose(0, 1)
            )
            # self.our_timer.time_and_clear(f'layer{i} bilinear')

        # print('v_li:', v_li)
        # print('pree-mlp out:',out)
        out = self.mlp(torch.cat(out, dim=1)).view(-1)
        # print('post-mlp out:',out)
        # self.our_timer.time_and_clear('final mlp')
        # self.our_timer.print_durations_log()
        # exit(-1)
        return out, X_q, X_t


class DynamicBilinear(torch.nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super(DynamicBilinear, self).__init__()
        self.weight = nn.Parameter(torch.empty((d_out, d_in, d_in)))
        if bias:
            self.bias = nn.Parameter(torch.empty((d_out)))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x, y):
        return \
            torch.matmul(
                torch.matmul(
                    x.view(1, *x.shape), self.weight
                ), y.view(1, *y.shape).transpose(-1, -2)
            ) + self.bias.view(-1, 1, 1)