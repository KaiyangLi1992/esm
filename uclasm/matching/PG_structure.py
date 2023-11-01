from itertools import product
import random
import networkx as nx
import numpy as np
from laptools import clap
from matching_utils import MonotoneArray
from datetime import datetime

def update_state(state,threshold):
    state.threshold = threshold
    changed_cands = np.ones((len(state.g1.nodes),), dtype=np.bool_)
    state.candidates = state.get_candidates()
                    # 从state_candidates中找出索引不在state_nn_mapping键中的行
    if np.any(np.all(state.candidates == False, axis=1)):
        return
    old_candidates = state.candidates.copy()
    state.localcosts[:] = np.maximum(state.localcosts, state.get_localcosts())
    state.globalcosts[:] = np.maximum(state.globalcosts, state.get_globalcosts())

    # TODO: Does this break if nodewise changes the candidates?
    while True:
        state.localcosts[:] = np.maximum(state.localcosts, state.get_localcosts())
        state.globalcosts[:] = np.maximum(state.globalcosts, state.get_globalcosts())
        state.candidates = state.get_candidates()
        if np.any(np.all(state.candidates == False, axis=1)):
            return

        changed_cands = np.any(state.candidates != old_candidates, axis=1)
        if ~np.any(changed_cands):
            break
        old_candidates = state.candidates.copy()



def swap_source_target(graph):
    pass
    # new_graph = nx.DiGraph()
    
    # for source, target in graph.edges():
    #     new_graph.add_edge(target, source)  # 交换源节点和目标节点
        
    # return new_graph
cache = {}
def get_adjacency_matrix_with_cache(graph):
    # 使用图的某种唯一标识符作为缓存键
    graph_id = graph.graph['gid']
    
    if graph_id not in cache:
        cache[graph_id] = nx.adjacency_matrix(graph)
        
    return cache[graph_id]

cache_T = {}
def get_adjacency_matrix_with_cache_T(graph):
    # 使用图的某种唯一标识符作为缓存键
    graph_id = graph.graph['gid']
    
    if graph_id not in cache_T:
        cache_T[graph_id] = nx.adjacency_matrix(graph).T
        
    return cache_T[graph_id]

def iter_adj_pairs(g1, g2,g1_reverse,g2_reverse):
   
    # start_time = datetime.now()
    yield (get_adjacency_matrix_with_cache(g1),get_adjacency_matrix_with_cache(g2))
    yield (get_adjacency_matrix_with_cache_T(g1),  get_adjacency_matrix_with_cache_T(g2))
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time}")



def get_non_matching_mask(state):
        """Gets a boolean mask for the costs array corresponding to all entries
        that would violate the matching."""
        mask = np.zeros((len(state.g1.nodes),len(state.g2.nodes)), dtype=np.bool_)
        matching = [(k, v) for k, v in state.nn_mapping.items()]
        if len(matching) > 0:
            mask[[pair[0] for pair in matching],:] = True
            mask[:,[pair[1] for pair in matching]] = True
            mask[tuple(np.array(matching).T)] = False
        return mask

class State(object):
    def __init__(self,g1,g2,threshold=np.inf,pruned_space=None,\
                 ori_candidates=None,g1_reverse=None,g2_reverse=None,nn_mapping={}):
        self.g1 = g1
        self.g2 = g2
        self.hn = None
        self.cn = None
        self.nn_mapping = nn_mapping

        if g1_reverse == None:
            self.g1_reverse = swap_source_target(g1)
        else:
            self.g1_reverse = g1_reverse

        if g2_reverse == None:
            self.g2_reverse = swap_source_target(g2)
        else:
            self.g2_reverse = g2_reverse

        if ori_candidates is None:
            self.ori_candidates = self.generate_ori_candidate()
        else:
            self.ori_candidates = ori_candidates
        
        if pruned_space is None:
            self.pruned_space = []
        else:
            self.pruned_space = pruned_space
        self.shape = (len(g1.nodes),len(g2.nodes))

        self.globalcosts = np.zeros(self.shape).view(MonotoneArray)
        self.localcosts = np.zeros(self.shape).view(MonotoneArray)

        self.threshold = threshold
        self.candidates = self.ori_candidates.copy()
        # self.localcosts = self.get_localcosts()
        # self.globalcosts = self.get_globalcosts()
        # self.candidates = self.get_candidates()
        # self.action_space = self.get_action_space()
        
        
    def get_action_heuristic(self):

        matrix = self.candidates
        exclude_indices = list(self.nn_mapping.keys())
        if not exclude_indices:
            filtered_matrix = matrix
        else:
            matrix[exclude_indices,:] = False
            filtered_matrix = matrix
        result_matrix =  np.where(matrix, self.globalcosts, np.inf)


        min_index = np.unravel_index(np.argmin(result_matrix), result_matrix.shape)
        min_index_tuple = tuple(min_index)

    
        return min_index_tuple
    
    def get_action_space(self,action_exp):

        matrix = self.candidates
        row_index = action_exp[0]
        row = matrix[row_index]

        # 获取非零元素的列坐标
        non_zero_columns = np.nonzero(row)[0]

        # 转换为 (x, y) 格式的坐标
        coordinates = [(row_index, col) for col in non_zero_columns]
        
        return coordinates
    
    def get_candidates(self):
        candidates =  (self.globalcosts < (self.threshold - 1e-8)).view(np.ndarray)
        candidates = np.logical_and(self.ori_candidates, candidates)
        rows = [tup[0] for tup in self.pruned_space]
        cols = [tup[1] for tup in self.pruned_space]
        candidates[rows, cols] = False
        return candidates

    
        
    def generate_ori_candidate(self):
        g1 = self.g1
        g2 = self.g2
        attrs_g1 = np.array([g1.nodes[i]['type'] for i in sorted(g1.nodes)])
        attrs_g2 = np.array([g2.nodes[i]['type'] for i in sorted(g2.nodes)])
        
        similarity_matrix = attrs_g1[:, None] == attrs_g2[None, :]
        
        return similarity_matrix
    
    def get_localcosts(self):
        g1 = self.g1
        g2 = self.g2
        g1_reverse = self.g1_reverse
        g2_reverse = self.g2_reverse
        candidates = self.candidates
        local_costs = np.zeros((len(g1.nodes),len(g2.nodes)))

        for dst_idx, src_idx in list(nx.Graph(g1).edges()):
            src_is_cand = candidates[src_idx]
            dst_is_cand = candidates[dst_idx]
            supported_edges = None

        
            total_tmplt_edges = 0
            for tmplt_adj, world_adj in iter_adj_pairs(g1, g2,g1_reverse,g2_reverse):
                tmplt_adj_val = tmplt_adj[src_idx, dst_idx]
                total_tmplt_edges += tmplt_adj_val

                # if there are no edges in this channel of the template, skip it
                if tmplt_adj_val == 0:
                    continue

                # sub adjacency matrix corresponding to edges from the source
                # candidates to the destination candidates
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
            local_costs[src_idx][src_is_cand] += src_least_cost

            if src_idx != dst_idx:
                dst_support = supported_edges.max(axis=0)
                dst_least_cost = total_tmplt_edges - dst_support.A
                dst_least_cost = np.array(dst_least_cost).flatten()
                local_costs[dst_idx][dst_is_cand] += dst_least_cost
        return local_costs
        
    def get_globalcosts(self):
        g1 = self.g1
        g2 = self.g2
        tmplt_idx_mask = np.ones(len(g1.nodes), dtype=np.bool_)
        world_idx_mask = np.ones(len(g2.nodes), dtype=np.bool_)


        global_costs = np.ones((len(g1.nodes),len(g2.nodes)))
        for tmplt_idx, world_idx in self.nn_mapping.items():
            tmplt_idx_mask[tmplt_idx] = False
            world_idx_mask[world_idx] = False
        # Prevent matches by setting their local cost to infinity
        # for tmplt_idx, world_idx in smp.prevented_matches:
        #     smp.local_costs[tmplt_idx, world_idx] = float("inf")
        local_costs = self.localcosts
        local_costs[~self.ori_candidates] = np.inf
        partial_match_cost = np.sum([local_costs[match]/2  for match in self.nn_mapping.items()])
        mask = np.ix_(tmplt_idx_mask, world_idx_mask)
        total_match_cost = partial_match_cost
        if np.any(tmplt_idx_mask):
            costs = local_costs[mask] / 2 
            global_cost_bounds = clap.costs(costs)
            global_costs[mask] = global_cost_bounds 
            total_match_cost += np.min(global_cost_bounds)
        non_matching_mask = get_non_matching_mask(self)
        global_costs[non_matching_mask] = float("inf")
        for tmplt_idx, world_idx in self.nn_mapping.items():
            global_costs[tmplt_idx, world_idx] = total_match_cost
        return global_costs


