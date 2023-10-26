import pickle
import os
import datetime
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import pickle
import numpy as np
import sys
import time

from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)

sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/") 
# from GraphGPS.graphgps.transform.posenc_stats import get_rw_landing_probs

# Custom module imports
# from NSUBS.model.OurSGM.config import FLAGS
# from NSUBS.model.OurSGM.saver import saver
# from NSUBS.model.OurSGM.train import train, cross_entropy_smooth
# from NSUBS.model.OurSGM.test import test
# from NSUBS.model.OurSGM.model_glsearch import GLS
# from NSUBS.model.OurSGM.utils_our import load_replace_flags
# from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
# from NSUBS.src.utils import OurTimer, save_pickle
# from environment import environment, update_state, calculate_cost
from torch import Tensor
from torch_geometric.utils import from_networkx

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))
    

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing



with open('/home/kli16/ISM_custom/esm_NSUBS/esm/data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05.pkl','rb') as f:
    dataset = pickle.load(f)
for graph in dataset.gs:
    nx_graph = graph.nxgraph
    data = from_networkx(nx_graph)
    skip_time = [i for  i in range(1,21) ]
    feature = get_rw_landing_probs(skip_time, data.edge_index)
    nx_graph.RWSE = feature


with open('/home/kli16/ISM_custom/esm_NSUBS/esm/data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05_RWSE.pkl','wb') as f:
    pickle.dump(dataset,f)