# from pynauty import *

# g = Graph(5)
#
# g.connect_vertex(0, [1, 2, 3])
#
#
# g.connect_vertex(2, [1, 3, 4])
#
# g.connect_vertex(4, [3])
#
# print(g)
#
#
# x = autgrp(g)
# print(x)


# g = Graph(6)
#
# g.connect_vertex(0, [1, 2, 3, 4, 5])
#
#
# print(g)
#
#
# x = autgrp(g)
# print(x)
#
# node_group_labels = x[3]
# num_groups = x[4]
# print('node_group_labels', node_group_labels)
# print('num_groups', num_groups)

from pynauty import Graph as PynautyGraph, autgrp

from config import FLAGS
from saver import saver
from data_loader import get_data_loader_wrapper
import networkx as nx
from train import train
from test import test
from model import DGMC, Encoder_GCN, Encoder_MLP
from model_glsearch import GLS
from utils_our import load_replace_flags
from utils import OurTimer

import torch
import traceback


def compute_automorphism_groups(g):
    nauty_g = PynautyGraph(g.number_of_nodes())
    for i, node in enumerate(sorted(g.nodes())):
        assert i == node, 'nxgraph is not 0-based consecutive integer indexed!'
        nauty_g.connect_vertex(node, list(g.neighbors(node)))
    x = autgrp(nauty_g)
    # print(x)
    #
    node_group_labels = x[3]
    num_groups = x[4]
    assert len(node_group_labels) == g.number_of_nodes()
    return node_group_labels, num_groups


def _assign_nauty_results_to_nxgraph(g, node_group_labels, num_groups):
    for i, node in enumerate(sorted(g.nodes())):
        assert i == node, 'nxgraph is not 0-based consecutive integer indexed!'
        assert 0 <= node_group_labels[i] < g.number_of_nodes()
        nx.set_node_attributes(g, {node: node_group_labels[i]}, name='auto_group')

    return g


def _automorphism(g, sp):
    node_group_labels, num_groups = compute_automorphism_groups(g)
    # print(node_group_labels, num_groups)
    g_save = _assign_nauty_results_to_nxgraph(g, node_group_labels, num_groups)
    nx.write_gexf(g_save, sp)


def main():
    test_loader = get_data_loader_wrapper('test')
    gtarget = None
    for i, gp in enumerate(test_loader):
        # print(gp)
        # node_group_labels, num_groups = compute_automorphism_groups(gp.gq.nx_graph)
        # print(node_group_labels, num_groups)
        # gq_save = _assign_nauty_results_to_nxgraph(gp.gq.nx_graph, node_group_labels, num_groups)
        _automorphism(gp.gq.nx_graph,
                      f'/home/anonymous/Documents/GraphMatching/model/OurSGM/file/gq_am_{FLAGS.test_dataset}_{i}.gexf')
        gtarget = gp.gt.nx_graph

        break
    print('@@@')
    timer = OurTimer()
    _automorphism(gtarget,
                  f'/home/anonymous/Documents/GraphMatching/model/OurSGM/file/gtarget_{FLAGS.test_dataset}.gexf')
    print(timer.time_and_clear(()))
    print('done')

if __name__ == '__main__':
    main()
