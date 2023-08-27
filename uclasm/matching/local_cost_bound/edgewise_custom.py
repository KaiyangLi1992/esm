"""Provide a function for bounding node assignment costs with edgewise info."""
import numpy as np
import pandas as pd
import numba
import os
import tqdm
import time
import networkx as nx
import scipy.sparse as sp

def edgewise_no_attrs(smp, changed_cands=None):
    """Compute edgewise costs in the case where no attribute distance function
    is provided.

    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem on which to compute edgewise cost bounds.
    changed_cands : ndarray(bool)
        Boolean array indicating which template nodes have smp.candidates that have
        changed since the last run of the edgewise filter. Only these nodes and
        their neighboring template nodes have to be reevaluated.
    """
    candidates = smp.candidates()
    new_local_costs =  np.zeros([smp.tmplt.number_of_nodes(),smp.world.number_of_nodes()])
    tmplt_adj = sp.csr_matrix(nx.adjacency_matrix(smp.tmplt))
    world_adj = sp.csr_matrix(nx.adjacency_matrix(smp.world))
    for src_idx, dst_idx in list(smp.tmplt.edges()):
        if changed_cands is not None:
            # If neither the source nor destination has changed, there is no
            # point in filtering on this pair of nodes
            if not (changed_cands[src_idx] or changed_cands[dst_idx]):
                continue

        # get indicators of candidate nodes in the world adjacency matrices
        src_is_cand = candidates[src_idx]
        dst_is_cand = candidates[dst_idx]
        if ~np.any(src_is_cand) or ~np.any(dst_is_cand):
            print("No smp.candidates for given nodes, skipping edge")
            continue

        # This sparse matrix stores the number of supported template edges
        # between each pair of smp.candidates for src and dst
        # i.e. the number of template edges between src and dst that also exist
        # between their smp.candidates in the world
        supported_edges = None

        # Number of total edges in the template between src and dst
        total_tmplt_edges = 0
        
        tmplt_adj_val = tmplt_adj[src_idx, dst_idx]
        total_tmplt_edges += tmplt_adj_val

        # if there are no edges in this channel of the template, skip it
        if tmplt_adj_val == 0:
            continue
        # sub adjacency matrix corresponding to edges from the source
        # smp.candidates to the destination smp.candidates
        world_sub_adj = world_adj[:, dst_is_cand][src_is_cand, :]

        # Edges are supported up to the number of edges in the template
        if supported_edges is None:
            supported_edges = world_sub_adj.minimum(tmplt_adj_val)
        else:
            supported_edges += world_sub_adj.minimum(tmplt_adj_val)

        src_support = supported_edges.max(axis=1)
        src_least_cost = total_tmplt_edges - src_support.A

        # Different algorithm from REU
        # Main idea: assigning u' to u and v' to v causes cost for u to increase
        # based on minimum between cost of v and missing edges between u and v
        # src_least_cost = np.maximum(total_tmplt_edges - supported_edges.A,
        #                             local_costs[dst_idx][dst_is_cand]).min(axis=1)

        src_least_cost = np.array(src_least_cost).flatten()
        # Update the local cost bound
        new_local_costs[src_idx][src_is_cand] += src_least_cost

        if src_idx != dst_idx:
            dst_support = supported_edges.max(axis=0)
            dst_least_cost = total_tmplt_edges - dst_support.A
            dst_least_cost = np.array(dst_least_cost).flatten()
            new_local_costs[dst_idx][dst_is_cand] += dst_least_cost
    smp.local_costs = new_local_costs
    return new_local_costs
