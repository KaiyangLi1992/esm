
from collections import defaultdict
from data_structures_search_tree_scalable import Bidomain, StateNode, ActionEdge, SearchTree,ActionSpaceData, unroll_bidomains, get_natts_hash, get_natts2g2abd_sg_nids
from data_structures_buffer_scalable import BinBuffer
from data_structures_common_scalable import StackHeap, DoubleDict, DQNInput
from layers_dqn_v1_scalable import Q_network_v1
from reward_calculator import RewardCalculator
from saver import saver
# from matching.matching_utils import MonotoneArray
from embedding_saver import EMBEDDING_SAVER
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from edgewise_custom import edgewise_no_attrs

class MonotoneArray(np.ndarray):
    """An ndarray whose entries cannot decrease.

    Example
    -------
    >>> A = np.zeros(3).view(MonotoneArray)
    >>> A[0:2] = [-1, 1]
    >>> A
    MonotoneArray([0.,  1.,  0.])

    """

    def __setitem__(self, key, value):
        """Ensure values cannot decrease."""
        value = np.maximum(self[key], value)
        super().__setitem__(key, value)


def create_candidates(g1,g2,natts2g2nids):
    candidates = np.zeros([len(g1.nodes),len(g2.nodes)],dtype = np.byte).view(MonotoneArray)
    for attr in natts2g2nids.keys():
        rows = list(natts2g2nids[attr]['g1'])
        colums = list(natts2g2nids[attr]['g2'])
        # candidates[rows, :][:, colums] = 1
        candidates[np.ix_(rows, colums)] = 1
    return candidates
    
    
def add_noise(G):
    
    # edges_to_remove = random.sample(G.edges(), int(0.05 * G.number_of_edges()))
    # G.remove_edges_from(edges_to_remove)

    # 添加约5%的新边
    edges_to_add = []
    for i in range(int(0.2 * G.number_of_edges())):
        source = random.randint(0, len(G.nodes)-1)
        target = random.randint(0, len(G.nodes)-1)
        while source == target or G.has_edge(source, target):
            source = random.randint(0, len(G.nodes)-1)
            target = random.randint(0, len(G.nodes)-1)
        edges_to_add.append((source, target))
    G.add_edges_from(edges_to_add)
    return G




if __name__ == "__main__":
    with open('/root/uclasm/toy_g1.pkl','rb') as f:
        g1 = pickle.load(f)
        g1 = add_noise(g1)
    with open('/root/uclasm/toy_g2.pkl','rb') as f:
        g2 = pickle.load(f)
    natts2g2nids = defaultdict(lambda: defaultdict(set))
    for nid in range(g1.number_of_nodes()):
        natts2g2nids[get_natts_hash(g1.nodes[nid])]['g1'].add(nid)
    for nid in range(g2.number_of_nodes()):
        natts2g2nids[get_natts_hash(g2.nodes[nid])]['g2'].add(nid)
    candidates = create_candidates(g1,g2,natts2g2nids)
    new_local_costs = edgewise_no_attrs(g1,g2, candidates)
    
    print(sum(sum(new_local_costs)))

