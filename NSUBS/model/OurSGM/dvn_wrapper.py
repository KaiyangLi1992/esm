from NSUBS.src.utils import OurTimer, graph_to_edge_tensor
import collections
import itertools
import torch
import networkx as nx
import numpy as np
from copy import deepcopy

from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.dvn_encoder import create_encoder
from NSUBS.model.OurSGM.dvn_decoder import create_decoder
from NSUBS.model.OurSGM.dvn_preencoder import create_preencoder
from NSUBS.model.OurSGM.dvn import DVN
from NSUBS.model.OurSGM.utils_nn import MLP, get_MLP_args
# from graphgps.network.gps_model import FeatureEncoder

def create_u2v_li(nn_map, cs_map, candidate_map):
    u2v_li = {}
    for u in cs_map.keys():
        if u in nn_map:
            v_li = [nn_map[u]]
        elif u in candidate_map:
            v_li = candidate_map[u]
        else:
            v_li = cs_map[u]
        u2v_li[u] = v_li
    return u2v_li

def create_dvn(d_in_raw, d_in):
    pre_encoder = create_preencoder(d_in_raw, d_in)
    # pre_encoder = FeatureEncoder(1)

    encoder_gnn_consensus, d_out = create_encoder(d_in)
    decoder_policy, decoder_value = create_decoder()
    mlp_final = MLP(*get_MLP_args([64, 32, 16, 8, 4, 1]))
    norm_li = \
        torch.nn.ModuleList([
            torch.nn.LayerNorm(d_in),
            torch.nn.LayerNorm(d_in),
            torch.nn.LayerNorm(d_out),
            torch.nn.LayerNorm(d_out),
        ])
    dvn = DVN(pre_encoder, encoder_gnn_consensus, decoder_policy, decoder_value, norm_li)
    dvn_wrapper = DVN_wrapper(dvn, mlp_final)
    return dvn_wrapper

class DVN_wrapper(torch.nn.Module):
    def __init__(self, dvn, mlp_final):
        super(DVN_wrapper, self).__init__()
        self.dvn = dvn
        self.mlp_final = mlp_final

    def reset_cache(self):
        self.dvn.encoder.reset_cache()

    def forward(self, gq, gt, u, v_li, nn_map, cs_map, candidate_map,
                cache_embeddings, graph_filter=None, filter_key=None,
                execute_action=None, query_tree=None):

        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()

        # unpack inputs
        Xq, edge_indexq, Xt, edge_indext,pyg_data_q,pyg_data_t = \
            gq.init_x.to(FLAGS.device), graph_to_edge_tensor(gq).to(FLAGS.device), gt.init_x.to(FLAGS.device),\
                  graph_to_edge_tensor(gt).to(FLAGS.device),gq.pyg_data.to(FLAGS.device),gt.pyg_data.to(FLAGS.device)
        u2v_li = create_u2v_li(nn_map, cs_map, candidate_map)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'create_u2v_li')

        # if graph_filter is None or len(nn_map) == 0:
        node_mask = np.array([])
        # else:
        #     node_mask, node_mask_inv = \
        #         graph_filter.get_node_mask(filter_key, gq.nx_graph, gt.nx_graph, nn_map.values(), u2v_li)
        #     edge_indext = \
        #         graph_filter.get_node_mask_oh_edge_index(edge_indext, node_mask_inv)#, gt.nx_graph.number_of_nodes())

        if FLAGS.time_analysis:
            timer.time_and_clear(f'node_mask {gt.nx_graph.number_of_nodes() - len(node_mask)} nodes selected; before {gt.nx_graph.number_of_nodes()}')

        out_policy, out_value, out_other = \
            self.dvn(
                Xq, edge_indexq, Xt, edge_indext,
                gq, gt,
                nn_map, cs_map, candidate_map,
                u2v_li, node_mask, cache_embeddings,
                execute_action, query_tree, pyg_data_q,pyg_data_t,u=u, v_li=v_li#,
            )
        if FLAGS.time_analysis:
            timer.time_and_clear(f'fast?')
            timer.print_durations_log()
        return out_policy, out_value, out_other

class EmbeddingCache:
    def __init__(self):
        self.Xt_li = []
        self.key = None

    def get_Xt_li(self, key):
        if key is None or key != self.key:
            return None
        else:
            return self.Xt_li

class GraphFilter:
    def __init__(self, cache_size=10):
        self.queue = collections.deque([None for _ in range(cache_size)], 100)
        self.cache = {}

    def add_to_cache(self, key, val):
        key_rm = self.queue.popleft()
        if key_rm is not None and key_rm != key:
            del self.cache[key_rm]
        if key not in self.cache:
            self.cache[key] = val
        self.queue.append(key)

    def check_if_new_filter(self, cur_node, nn_map):
        if len(nn_map) == 1:
            filter_key = self.get_filter_key(nn_map)
        else:
            filter_key = cur_node.filter_key
        return filter_key

    def get_filter_key(self, nn_map):
        filter_key = frozenset(nn_map.values())
        return filter_key

    def get_node_mask_diameter(self, gq, gt, root_li):
        assert len(root_li) >= 1
        k = max(nx.diameter(gq), 2)
        sel_nodes = set()
        for root in root_li:
            if len(sel_nodes) == 0:
                # sel_nodes = set(nx.ego_graph(gt, root, k).nodes())
                sel_nodes = set([v for _, v_successors in nx.bfs_successors(gt, root, k) for v in v_successors]).union({root})
            else:
                sel_nodes = sel_nodes.intersection(nx.ego_graph(gt, root, k).nodes())
        return sel_nodes

    def get_node_mask_circle(self, u2v_li):
        sel_nodes = set(itertools.chain.from_iterable([v_li for v_li in u2v_li.values()]))
        return sel_nodes

    def get_node_mask(self, filter_key, gq, gt, root_li, u2v_li):
        if FLAGS.use_node_mask_diameter:
            # import time
            if filter_key is not None and filter_key in self.cache:
                node_mask_diameter = self.cache[filter_key]
                # t0 = time.time()
                # t1 = t0
            else:
                # t0 = time.time()
                node_mask_diameter = self.get_node_mask_diameter(gq, gt, root_li)
                self.add_to_cache(filter_key, node_mask_diameter)
                # t1 = time.time()
        else:
            node_mask_diameter = None
        node_mask_circle = self.get_node_mask_circle(u2v_li)
        # t2 = time.time()
        node_mask_inv = \
             node_mask_circle if node_mask_diameter is None else \
                 node_mask_diameter.intersection(node_mask_circle)
        node_mask = set(gt.nodes()) - node_mask_inv
        # t3 = time.time()
        # print(f'diameter:{t1-t0}\tcircle:{t2-t1}\tmerge:{t3-t2}\tfilter_key:{filter_key}')
        return list(node_mask), list(node_mask_inv)

    def get_node_mask_oh_edge_index(self, edge_index, node_mask):#, gt_num_nodes):
        edge_index_np = edge_index.detach().cpu().numpy()
        edge_index_filtered = edge_index[:, np.in1d(edge_index_np[0], node_mask)]# * np.in1d(edge_index_np[1], node_mask)]

        return edge_index_filtered # node_mask_one_hot,


