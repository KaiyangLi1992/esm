
"""Provides a method for performing a greedy search for solutions using
total costs."""

import numpy as np
import bisect
import math

import networkx as nx

from .data_structures_search_tree import SearchTree, ActionEdge
from .search_utils import *
from global_cost_bound import *
from local_cost_bound import *
from matching_problem import MatchingProblem
from utils import one_hot
from heapq import heappush, heappop,heapify

def get_reward(tmplt_idx,cand_idx,smp,state):
    tmplt_nx = smp.tmplt
    cand_nx = smp.world
    neighbors = set([n for n in tmplt_nx[tmplt_idx]])
    tmplt_matched_nodes = set([n for n in state.matching_dict.keys()])
    tmplt_node_intersection  = neighbors.intersection(tmplt_matched_nodes)
    cand_node_intersection = set([state.matching_dict[n] for n in list(tmplt_node_intersection)])
    neighbors_cand = set([n for n in cand_nx[cand_idx]])

    posi_reward = len(list(neighbors_cand.intersection(cand_node_intersection)))
    nega_reward = len(list(tmplt_node_intersection)) - posi_reward 
    reward1 = posi_reward - nega_reward


    tmplt_nx = nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(tmplt_nx).T)
    cand_nx = nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(cand_nx).T)
    neighbors = set([n for n in tmplt_nx[tmplt_idx]])
    tmplt_matched_nodes = set([n for n in state.matching_dict.keys()])
    tmplt_node_intersection  = neighbors.intersection(tmplt_matched_nodes)
    cand_node_intersection = set([state.matching_dict[n] for n in list(tmplt_node_intersection)])
    neighbors_cand = set([n for n in cand_nx[cand_idx]])

    posi_reward = len(list(neighbors_cand.intersection(cand_node_intersection)))
    nega_reward = len(list(tmplt_node_intersection)) - posi_reward 
    reward2 = posi_reward - nega_reward



    return reward1+reward2
def greedy_best_k_matching_custom(smp, k=1, nodewise=True, edgewise=True,
                           verbose=True):
    # if smp.global_cost_threshold == float("inf"):
    #     raise Exception("Invalid global cost threshold.")

    # States still left to be processed
    open_list = []
    # States which have already been processed
    closed_list = []
    # Map from states to computed costs
    cost_map = {}
    # States where all nodes have been assigned
    solutions = []

    # Map from template indexes to world indexes
    current_matching = {}

    # Initialize matching with known matches
    candidates = smp.candidates()
    for i in range(smp.tmplt.number_of_nodes()):
        if np.sum(candidates[i]) == 1:
            current_matching[i] = np.argwhere(candidates[i])[0][0]

    start_state = State()
    start_state.matching_dict = current_matching
    start_state.matching = tuple_from_dict(current_matching)
    start_state.cost = smp.global_costs.min()
    search_tree = SearchTree(start_state)
    cost_map[start_state.matching] = start_state.cost

    # Handle the case where we start in a solved state
    if len(start_state.matching) == smp.tmplt.number_of_nodes():
        solutions.append(start_state)
        return solutions

    heappush(open_list, start_state)


    # Cost of the kth solution
    kth_cost = float("inf")
    step = 0

    while len(open_list) > 0:

        current_state = heappop(open_list)
        # Ignore states whose cost is too high
        if current_state.cost > smp.global_cost_threshold or current_state.cost >= kth_cost:
            # Only print multiples of 10000 for skipped states
            if verbose and len(open_list) % 10000 == 0:
                print("Skipped state: {} matches".format(len(current_state.matching)),
                      "{} open states".format(len(open_list)), "current_cost:", current_state.cost,
                      "kth cost:", kth_cost, "max cost", smp.global_cost_threshold, "solutions found:", len(solutions))
            continue
        # if verbose and current_state.cost == kth_cost:
        if step%100==0:
            print(step)
            print("Current state: {} matches".format(len(current_state.matching)),
                  "{} open states".format(len(open_list)), "current_cost:", current_state.cost,
                  "kth cost:", kth_cost, "max cost", smp.global_cost_threshold, "solutions found:", len(solutions))
        step += 1
        curr_smp = smp.copy()
        curr_smp.enforce_matching(current_state.matching)
        # Do not reduce world as it can mess up the world indices in the matching
        iterate_to_convergence(curr_smp, reduce_world=False, nodewise=nodewise,
                               edgewise=edgewise)
        matching_dict = dict_from_tuple(current_state.matching)
        candidates = curr_smp.candidates()
        # Identify template node with the least number of candidates
        cand_counts = np.sum(candidates, axis=1)
        # Prevent previously matched template idxs from being chosen
        cand_counts[list(matching_dict)] = np.max(cand_counts) + 1
        tmplt_idx = np.argmin(cand_counts)
        cand_idxs = np.argwhere(candidates[tmplt_idx]).flatten()
        # if verbose:
        #     print("Choosing candidate for", tmplt_idx,
        #           "with {} possibilities".format(len(cand_idxs)))

        # Only push states that have a total cost bound lower than the threshold
        for cand_idx in cand_idxs:
            new_matching = matching_dict.copy()
            new_matching[tmplt_idx] = cand_idx
            new_matching_tuple = tuple_from_dict(new_matching)
            if new_matching_tuple not in cost_map:
                new_state = State()
                new_state.matching_dict = new_matching 
                new_state.matching = new_matching_tuple
                new_state.cost = curr_smp.global_costs[tmplt_idx, cand_idx]
                if new_state.cost > smp.global_cost_threshold or new_state.cost >= kth_cost:
                    continue
                cost_map[new_matching_tuple] = new_state.cost
                if len(new_state.matching) == smp.tmplt.number_of_nodes():
                    solutions.append(new_state)
                    if k > 0 and len(solutions) > k:
                        solutions.sort()
                        solutions.pop()
                        heapify(solutions)
                        kth_cost = max(solutions).cost
                        smp.global_cost_threshold = min(smp.global_cost_threshold,
                                                        kth_cost)
                        iterate_to_convergence(smp, reduce_world=False, nodewise=nodewise,
                                                   edgewise=edgewise)
                else:
                    heappush(open_list, new_state)
                    reward = get_reward(tmplt_idx,cand_idx,smp,current_state)
                    new_state.cum_reward = current_state.cum_reward + reward
                    # print(current_state.matching)
                    # print((tmplt_idx,cand_idx))
                    # print(reward)
                    action_edge = ActionEdge((tmplt_idx,cand_idx),reward)
                    search_tree.link_states(
                        current_state, action_edge, new_state, 0, 0)
            else:
                if verbose:
                    print("Recognized state: ", new_matching)
    if verbose and len(solutions) < 100:
        for solution in solutions:
            solution_ob = {}
            for mapping in solution.matching:
                solution_ob[smp.tmplt.nodes[mapping[0]]]= smp.world.nodes[mapping[1]]
            print(str(solution_ob) + ": " + str(solution.cost))
    return search_tree