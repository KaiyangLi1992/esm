
"""Provides a method for performing a greedy search for solutions using
total costs."""

import numpy as np
import bisect
import math
import time
import networkx as nx
import time
from .data_structures_search_tree  import SearchTree, ActionEdge,get_natts_hash,Bidomain
from .search_utils import *
from global_cost_bound import *
from local_cost_bound import *
from matching_problem import MatchingProblem
from utils import one_hot
from heapq import heappush, heappop,heapify
from collections import defaultdict
from batch import create_edge_index,create_adj_set

def swap_source_target(graph):
    new_graph = nx.DiGraph()
    
    for source, target in graph.edges():
        new_graph.add_edge(target, source)  # 交换源节点和目标节点
        
    return new_graph
def has_self_loop(graph, node):
    return node in graph.successors(node) 

def get_init_reward(nodes,graph):
    subgraph = graph.subgraph(nodes)
    subgraph_edge_count = subgraph.number_of_edges()
    return subgraph_edge_count

def update_bidomains(g1, g2, action, nn_map, nn_map_neighbors, natts2bds, natts2g2nids):

    natts2bds_new = defaultdict(list)

    '''
        1*)
            Ni = (N - (N1 + N2 + ... +Ni-1)).intersect(abdi,j)
                Ni == N.intersect(abd)
            abd1*,j = Ni - action                        <bitstring = <bitstringi>, 1> s.t. j is label
            abd0*,j = abdi,j - abd1*,j - action           <bitstring = <bitstringi>, 0>
                abd2,j == abd - abd1,j - action
    '''
    u, v = action
    N_g1 = set(g1.neighbors(u)) - set(nn_map.keys())
    N_g2 = set(g2.neighbors(v)) - set(nn_map.values())

    nn_map_neighbors_new =\
        {
            'g1': nn_map_neighbors['g1'].union(N_g1) - {u},
            'g2': nn_map_neighbors['g2'].union(N_g2) - {v}
        }

    for natts, bidomains in natts2bds.items():
        for bidomain in bidomains:
            assert natts == bidomain.natts
            left, right = bidomain.left, bidomain.right
            Ni_g1, Ni_g2 = N_g1.intersection(left), N_g2.intersection(right)
            left_1, right_1 = Ni_g1 - {u}, Ni_g2 - {v}
            left_0, right_0 = left - Ni_g1 - {u}, right - Ni_g2 - {v}
            if len(left_1) > 0 and len(right_1) > 0:
                natts2bds_new[natts].append(
                    Bidomain(left_1, right_1, natts))
            if len(left_0) > 0 and len(right_0) > 0:
                natts2bds_new[natts].append(
                    Bidomain(left_0, right_0, natts))

            # remaining nodes will not belong to any adjacent bidomain!
            # => partition from unadjacent bidomain
            N_g1, N_g2 = N_g1 - Ni_g1, N_g2 - Ni_g2

    '''
        2*)
            let N(i) = neighbors with label i
            abd1,i
            = unconnected[N(i) - (N1, N2, ..., Nk)] - action
            = unconnected[(N(i) - (N1, N2, ..., Nk))(i)] - action    <bitstring = 0, ..., 0, 1_i> s.t. i is label
    '''
    for natts, g2nids in natts2g2nids.items():
        nid_natts_all_g1, nid_natts_all_g2 = g2nids['g1'], g2nids['g2']
        left_1_natts = N_g1.intersection(nid_natts_all_g1) - {u} - nn_map_neighbors['g1']
        right_1_natts = N_g2.intersection(nid_natts_all_g2) - {v} - nn_map_neighbors['g2']
        if len(left_1_natts) > 0 and len(right_1_natts) > 0:
            natts2bds_new[natts].append(
                Bidomain(left_1_natts, right_1_natts, natts))
    return natts2bds_new, nn_map_neighbors_new
    
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

    start_time = time.time()
    # tmplt_nx = nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(tmplt_nx).T)
    # cand_nx = nx.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(cand_nx).T)
    tmplt_nx = smp.reverse_tmplt 
    cand_nx = smp.reverse_world 
    end_time = time.time()
    execution_time = end_time - start_time
    neighbors = set([n for n in tmplt_nx[tmplt_idx]])
    tmplt_matched_nodes = set([n for n in state.matching_dict.keys()])
    tmplt_node_intersection  = neighbors.intersection(tmplt_matched_nodes)
    cand_node_intersection = set([state.matching_dict[n] for n in list(tmplt_node_intersection)])
    neighbors_cand = set([n for n in cand_nx[cand_idx]])

    posi_reward = len(list(neighbors_cand.intersection(cand_node_intersection)))
    nega_reward = len(list(tmplt_node_intersection)) - posi_reward 
    reward2 = posi_reward - nega_reward
    cycle_reward = 0
    if has_self_loop(tmplt_nx, tmplt_idx) & has_self_loop(cand_nx, cand_idx):
        cycle_reward = 1
    return reward1+reward2+cycle_reward

def greedy_best_k_matching_custom(smp,state_init, k=1, nodewise=True, edgewise=True,verbose=True):
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
    candidates = smp.get_candidates()
    # for i in range(smp.tmplt.number_of_nodes()):
    #     if np.sum(candidates[i]) == 1:
    #         current_matching[i] = np.argwhere(candidates[i])[0][0]

    g1 = smp.tmplt
    g2 = smp.world
    # natts2g2nids = defaultdict(lambda: defaultdict(set))
    # for nid in range(g1.number_of_nodes()):
    #     natts2g2nids[get_natts_hash(g1.nodes[nid])]['g1'].add(nid)
    # for nid in range(g2.number_of_nodes()):
    #     natts2g2nids[get_natts_hash(g2.nodes[nid])]['g2'].add(nid)

    # edge_index1, edge_index2 = \
    #             create_edge_index(g1), create_edge_index(g2)
    # adj_list1, adj_list2 = \
    #             create_adj_set(g1), create_adj_set(g2)

    start_state = State(natts2g2nids=state_init.natts2g2nids,g1=smp.tmplt,g2=smp.world,
                        edge_index1=state_init.edge_index1,edge_index2=state_init.edge_index2,
                        adj_list1=state_init.adj_list1,adj_list2=state_init.adj_list2,
                        ins_g1=state_init.ins_g1,ins_g2=state_init.ins_g2,cur_id=state_init.cur_id)
    
    start_state.matching_dict = current_matching
    start_state.matching = tuple_from_dict(current_matching)
    start_state.cost = smp.global_costs.min()
    matched_nodes = [nodes for nodes in current_matching.keys()]
    start_state.cum_reward = get_init_reward(matched_nodes,smp.tmplt)
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
        print(len(open_list))
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
        start_time =  time.time()
        iterate_to_convergence(curr_smp, reduce_world=False, nodewise=nodewise,
                               edgewise=edgewise)
        end_time = time.time()
        execution_time = end_time - start_time
        matching_dict = dict_from_tuple(current_state.matching)
        candidates = curr_smp.get_candidates()
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
                new_state = State(natts2g2nids = state_init.natts2g2nids,g1=smp.tmplt,g2=smp.world,
                                  edge_index1=current_state.edge_index1,edge_index2=current_state.edge_index2,
                                  adj_list1=current_state.adj_list1,adj_list2=current_state.adj_list2,
                                  ins_g1=current_state.ins_g1,ins_g2=current_state.ins_g2,
                                  cur_id=current_state.cur_id)
                new_state.matching_dict = new_matching 
                new_state.matching = new_matching_tuple
                new_state.cost = curr_smp.global_costs[tmplt_idx, cand_idx]
                if new_state.cost > smp.global_cost_threshold or new_state.cost >= kth_cost:
                    continue

                action = (tmplt_idx,cand_idx)
                
                natts2bds, nn_map_neighbors =  update_bidomains(g1, g2, action, 
                                                                current_state.matching_dict, current_state.nn_map_neighbors, 
                                                                current_state.natts2bds,
                                                                current_state.natts2g2nids)
                new_state.natts2bds = natts2bds
                new_state.nn_map_neighbors = nn_map_neighbors

                cost_map[new_matching_tuple] = new_state.cost
                reward = get_reward(tmplt_idx,cand_idx,smp,current_state)
                action_edge = ActionEdge((tmplt_idx,cand_idx),reward)
                search_tree.link_states(
                        current_state, action_edge, new_state, 0, 1)


                if len(new_state.matching) == smp.tmplt.number_of_nodes():
                    solutions.append(new_state)
                    if k > 0 and len(solutions) > k:
                        solutions.sort()
                        solutions.pop()
                        heapify(solutions)
                        kth_cost = max(solutions).cost
                        smp.global_cost_threshold = min(smp.global_cost_threshold,
                                                        kth_cost)
                        start_time =  time.time()
                        iterate_to_convergence(smp, reduce_world=False, nodewise=nodewise,
                                                   edgewise=edgewise)
                        
                else:
                    heappush(open_list, new_state)
                   
                    
                    # end_time = time.time()
                    # execution_time = end_time - start_time
                    # new_state.cum_reward = current_state.cum_reward + reward
                    # print(current_state.matching)
                    # print((tmplt_idx,cand_idx))
                    # print(reward)
                    
            else:
                if verbose:
                    print("Recognized state: ", new_matching)
    if verbose and len(solutions) < 100:
        for solution in solutions:
        #     solution_ob = {}
        #     for mapping in solution.matching:
        #         solution_ob[smp.tmplt.nodes[mapping[0]]]= smp.world.nodes[mapping[1]]
            print("Final cost: " + str(solution.cost))
    return search_tree,None