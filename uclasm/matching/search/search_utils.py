"""Utility functions and classes for search"""
import numpy as np
from collections import defaultdict
from .data_structures_search_tree import Bidomain
import global_cost_bound
import local_cost_bound
def assign_bids(natts2bds):
    # to ensure that given the same natts2bds, bid assignment is deterministic => sorted
    bid = 0
    for natts in sorted(natts2bds.keys()):
        for bd in natts2bds[natts]:
            bd.bid = bid
            bid += 1
class State:
    """A state for the greedy search algorithm.
    Attributes
    ----------
    matching : tuple
        Tuple representation of the current matching.
    cost: float
        Estimated cost of the matching.
    """
    def __init__(self,g1,g2,natts2g2nids,edge_index1,edge_index2,adj_list1,adj_list2,ins_g1,ins_g2,cur_id,
                 natts2bds={},nn_map_neighbors = {'g1': set(), 'g2': set()},cum_reward=0,num_steps=0,):
        self.matching = None
        self.matching_dict = None
        self.action_next_list = []
        self.cost = float("inf")
        self.cum_reward = cum_reward
        self.num_steps = num_steps
        self.v_search_tree = 0  # exhausted_q_max_LB
        self.natts2bds = natts2bds
        self.natts2g2nids = natts2g2nids
        self.nn_map_neighbors = nn_map_neighbors
        self.exhausted_v = set() 
        self.exhausted_w = set() 
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.adj_list1 = adj_list1
        self.adj_list2 = adj_list2
        self.g1 = g1
        self.g2 = g2
        self.ins_g1 = ins_g1
        self.ins_g2 = ins_g2
        self.cur_id = cur_id
        
    
        

    def __lt__(self, other):
        # TODO: Is this function the source of your sorting related time expenditures?
        if len(self.matching) != len(other.matching):
            return len(self.matching) > len(other.matching)
        return self.cost < other.cost

    def __str__(self):
        return str(self.matching) + ": " + str(self.cost)
    def assign_v(self, discount):
        # unrolling the recursive function calls
        state_stack_order = [self]
        state_stack_compute = []
        while len(state_stack_order) != 0:
            state_popped = state_stack_order.pop()
            state_stack_compute.append(state_popped)
            for action_next in state_popped.action_next_list:
                state_stack_order.append(action_next.state_next)

        # recursively compute LB
        for state in state_stack_compute[::-1]:
            for action_next in state.action_next_list:
                v_max_next_state = action_next.state_next.v_search_tree
                q_max_cur_state = \
                    action_next.reward + discount * v_max_next_state
                state.v_search_tree = max(state.v_search_tree, q_max_cur_state)
    def get_natts2bds_unexhausted(self, with_bids=True):
        if len(self.matching_dict) == 0:
            natts2bds_unexhausted = self.get_natts2bds_ubd_unexhausted()
        else:
            natts2bds_unexhausted = self.get_natts2bds_abd_unexhausted()
        if with_bids:
            assign_bids(natts2bds_unexhausted)
        return natts2bds_unexhausted
    def get_natts2bds_abd_unexhausted(self):
        natts2bds_unexhausted = defaultdict(list)
        for natts, bds in self.natts2bds.items():
            for bd in bds:
                assert bd.natts == natts
                if len(bd.left.intersection(self.exhausted_v)) > 0 or \
                        len(bd.right.intersection(self.exhausted_w)) > 0:
                    left = bd.left - self.exhausted_v
                    right = bd.right - self.exhausted_w
                    if len(left) > 0 and len(right) > 0:
                        natts2bds_unexhausted[natts].append(
                            Bidomain(left, right, natts)
                        )
                else:
                    assert len(bd.left) > 0 and len(bd.right) > 0
                    natts2bds_unexhausted[natts].append(bd)
        return natts2bds_unexhausted
    def get_natts2bds_ubd_unexhausted(self):
        natts2bds_unexhausted = defaultdict(list)
        for natts, g2nids in self.natts2g2nids.items():
            left = g2nids['g1'] - self.exhausted_v
            right = g2nids['g2'] - self.exhausted_w
            if len(left) > 0 and len(right) > 0:
                natts2bds_unexhausted[natts].append(
                    Bidomain(left, right, natts)
                )
        return natts2bds_unexhausted

def tuple_from_dict(dict):
    """Turns a dict into a representative sorted tuple of 2-tuples.
    Parameters
    ----------
    dict: dict
        A dictionary.
    Returns
    -------
    tuple
        A unique tuple representation of the dictionary as a sequence of
        2-tuples of key-value pairs, sorted by the dictionary key.
    """
    return tuple(sorted(dict.items()))

def dict_from_tuple(tuple):
    """Turns a tuple of 2-tuples into a dict.
    Parameters
    ----------
    tuple: tuple
        A unique tuple representation of the dictionary as a sequence of
        2-tuples of key-value pairs. Keys must be unique.
    Returns
    -------
    dict
        The equivalent dictionary.
    """
    return dict(tuple)

# This function is now deprecated, use MatchingProblem.enforce_matching instead
def set_fixed_costs(fixed_costs, matching):
    """Set fixed costs to float("inf") to enforce the given matching."""
    mask = np.zeros(fixed_costs.shape, dtype=np.bool)
    mask[[pair[0] for pair in matching],:] = True
    mask[:,[pair[1] for pair in matching]] = True
    mask[tuple(np.array(matching).T)] = False
    fixed_costs[mask] = float("inf")

import tqdm

def add_node_attr_costs(smp, node_attr_fn):
    """Increase the fixed costs to account for difference in node attributes."""
    tmplt_attr_keys = [attr for attr in smp.tmplt.nodelist.columns]
    tmplt_attr_cols = [smp.tmplt.nodelist[key] for key in tmplt_attr_keys]
    tmplt_attrs_zip = zip(*tmplt_attr_cols)
    world_attr_keys = [attr for attr in smp.world.nodelist.columns]
    world_attr_cols = [smp.world.nodelist[key] for key in world_attr_keys]
    with tqdm.tqdm(total=smp.tmplt.n_nodes * smp.world.n_nodes, ascii=True) as pbar:
        # for tmplt_idx, tmplt_row in smp.tmplt.nodelist.iterrows():
        for tmplt_idx, tmplt_attrs in enumerate(tmplt_attrs_zip):
            # for world_idx, world_row in smp.world.nodelist.iterrows():
            world_attrs_zip = zip(*world_attr_cols)
            for world_idx, world_attrs in enumerate(world_attrs_zip):
                pbar.update(1)
                if smp.fixed_costs[tmplt_idx, world_idx] != float("inf"):
                    tmplt_row = dict(zip(tmplt_attr_keys, tmplt_attrs))
                    world_row = dict(zip(world_attr_keys, world_attrs))
                    smp.fixed_costs[tmplt_idx, world_idx] += node_attr_fn(tmplt_row, world_row)

def add_node_attr_costs_identity(smp):
    """Assume node attr fn is the sum of the difference between attributes."""
    world_nodelist_np = np.array(smp.world.nodelist)
    for tmplt_idx, tmplt_row in smp.tmplt.nodelist.iterrows():
        tmplt_row_np = np.array(tmplt_row)
        # Remove first column: node ID which shouldn't be checked
        # Index to remove empty attributes
        nonempty_attrs = tmplt_row[1:] != ""
        smp.fixed_costs[tmplt_idx] += (tmplt_row_np[None, 1:][:,nonempty_attrs] != world_nodelist_np[:,1:][:,nonempty_attrs]).sum(axis=1)

def iterate_to_convergence(smp, reduce_world=True, nodewise=False,
                           edgewise=True, changed_cands=None, verbose=False):
    """Iterates the various cost bounds until the costs converge.
    Parameters
    ----------
    smp : MatchingProblem
        A subgraph matching problem to iterate cost bounds on until convergence
    reduce_world : bool
        Option to reduce the world by removing world nodes that are not
        candidates for any template node.
    changed_cands : np.ndarray(bool)
        Array of boolean values indicating which candidate nodes have changed
        candidates since the last time filters were run.
    verbose : bool
        Flag for verbose output.
    """
    if changed_cands is None:
        changed_cands = np.ones((smp.tmplt.number_of_nodes(),), dtype=np.bool)

    old_candidates = smp.get_candidates().copy()
    if smp._local_costs is not None:
        global_cost_bound.from_local_bounds(smp)

    # TODO: Does this break if nodewise changes the candidates?
    while True:
        if nodewise:
            if verbose:
                print(smp)
                print("Running nodewise cost bound")
            local_cost_bound.nodewise(smp)
            global_cost_bound.from_local_bounds(smp)
        if edgewise:
            if verbose:
                print(smp)
                print("Running edgewise cost bound")
            # TODO: does changed_cands work for edgewise?
            # local_cost_bound.edgewise(smp, changed_cands=changed_cands)
            local_cost_bound.edgewise(smp)
            global_cost_bound.from_local_bounds(smp)
        candidates = smp.get_candidates()
        if ~np.any(candidates):
            break
        changed_cands = np.any(candidates != old_candidates, axis=1)
        if ~np.any(changed_cands):
            break
        if reduce_world:
            smp.reduce_world()
        old_candidates = smp.get_candidates().copy()
        if smp.match_fixed_costs:
            # Remove non-candidates permanently by setting fixed costs to infinity
            non_cand_mask = np.ones(smp.shape, dtype=np.bool)
            non_cand_mask[old_candidates] = False
            smp.fixed_costs[non_cand_mask] = float("inf")
    if verbose:
        print(smp)
