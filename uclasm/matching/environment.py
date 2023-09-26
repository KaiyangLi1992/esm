
import pickle
import sys 
import networkx as nx
import torch
sys.path.append("/home/kli16/ISM_custom/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm/rlmodel") 
sys.path.append("/home/kli16/ISM_custom/esm/uclasm/") 
import numpy as np
from matching.search.data_structures_search_tree import SearchTree
from matching.PG_structure import State,update_state
from torch.utils.data import DataLoader
from collections import Counter
import random
def has_self_loop(graph, node):
    return node in graph.successors(node) 

def get_reward(tmplt_idx,cand_idx,state):
    g1 = state.g1
    g2 = state.g2
    g1_reverse = state.g1_reverse
    g2_reverse = state.g2_reverse

    tmplt_nx = g1
    cand_nx = g2
    neighbors = set([n for n in tmplt_nx[tmplt_idx]])
    tmplt_matched_nodes = set([n for n in state.nn_mapping.keys()])
    tmplt_node_intersection  = neighbors.intersection(tmplt_matched_nodes)
    cand_node_intersection = set([state.nn_mapping[n] for n in list(tmplt_node_intersection)])
    neighbors_cand = set([n for n in cand_nx[cand_idx]])

    posi_reward = len(list(neighbors_cand.intersection(cand_node_intersection)))
    nega_reward = len(list(tmplt_node_intersection)) - posi_reward 
    reward1 = posi_reward - nega_reward

    tmplt_nx = g1_reverse
    cand_nx = g2_reverse
    neighbors = set([n for n in tmplt_nx[tmplt_idx]])
    tmplt_matched_nodes = set([n for n in state.nn_mapping.keys()])
    tmplt_node_intersection  = neighbors.intersection(tmplt_matched_nodes)
    cand_node_intersection = set([state.nn_mapping[n] for n in list(tmplt_node_intersection)])
    neighbors_cand = set([n for n in cand_nx[cand_idx]])

    posi_reward = len(list(neighbors_cand.intersection(cand_node_intersection)))
    nega_reward = len(list(tmplt_node_intersection)) - posi_reward 
    reward2 = posi_reward - nega_reward
    cycle_reward = 0
    if has_self_loop(tmplt_nx, tmplt_idx) & has_self_loop(cand_nx, cand_idx):
        cycle_reward = 1
    return reward1+reward2+cycle_reward

def shuttle_node_id(G):
    nodes = list(G.nodes())
    random.shuffle(nodes)

    # 创建一个映射，将原始节点映射到新的随机节点
    mapping = {original: new for original, new in zip(G.nodes(), nodes)}

    # 使用映射创建一个新的DiGraph
    H = nx.relabel_nodes(G, mapping)
    return H

def get_attr_dict(G):
    type_dict = {}

    # 遍历所有节点
    for node in G.nodes(data=True):
        node_name = node[0]
        node_type = node[1]['type']
        
        # 将节点名字添加到对应的类型列表中
        if node_type not in type_dict:
            type_dict[node_type] = set()
        type_dict[node_type].add(node_name)
    return type_dict


def get_next_item(data_loader):
    data_iter = iter(data_loader)
    def helper():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            return next(data_iter)
    return helper()


def get_init_action(lst):
        first_elements = [t[0] for t in lst]

        # 计算每个元素的出现次数
        element_counts = Counter(first_elements)

        # 找出出现次数最少的元素
        min_count = min(element_counts.values())
        min_count_elements = [key for key, value in element_counts.items() if value == min_count]

        # 从最少出现的元素中随机选择一个
        chosen_element = random.choice(min_count_elements)

        # 从列表中选择所有元组，其第一个元素是chosen_element
        valid_tuples = [t for t in lst if t[0] == chosen_element]

        # 从valid_tuples中随机选择一个元组
        chosen_tuple = random.choice(valid_tuples)
        return chosen_tuple
def calculate_cost(small_graph, big_graph, mapping):
    cost = 0
    for edge in small_graph.edges():
        # 根据映射找到大图中的对应节点
        mapped_edge = (mapping[edge[0]], mapping[edge[1]])
        
        # 检查对应的边是否在大图中
        if not big_graph.has_edge(*mapped_edge):
            cost += 1

    return cost



class environment:
    def __init__(self,dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.searchtree = None
        self.g1 = None
        self.g2 = None
        self.threshold = np.inf
        
    def reset(self):
       batch_gids = get_next_item(self.data_loader)
    #    batch_gids = [torch.tensor([1]), torch.tensor([0])]
       self.g1 = self.dataset.look_up_graph_by_gid(batch_gids[0][0].item()).get_nxgraph()
       self.g2 = self.dataset.look_up_graph_by_gid(batch_gids[1][0].item()).get_nxgraph()

       self_loops = [(u, v) for u, v in self.g1.edges() if u == v]
       self.g1.remove_edges_from(self_loops)


    #    self.g1 = shuttle_node_id(self.g1)


       self.g1.attr_dict =  get_attr_dict(self.g1)
       self.g2.attr_dict =  get_attr_dict(self.g2)
       state_init = State(self.g1,self.g2)
       self.threshold = np.inf
       return state_init
    
    def step(self,state,action):
        nn_mapping = state.nn_mapping.copy()
        nn_mapping[action[0]] = action[1]
        state.pruned_space.append((action[0],action[1]))
        new_state = State(state.g1,state.g2,
                          nn_mapping=nn_mapping,
                          g1_reverse=state.g1_reverse,
                          g2_reverse=state.g2_reverse,
                          ori_candidates=state.ori_candidates)
        reward = get_reward(action[0],action[1],state)
        if len(nn_mapping) == len(state.g1.nodes):
            update_state(new_state,self.threshold)
            cost = new_state.globalcosts.min()
            if cost < self.threshold:
                self.threshold = cost
                return new_state,state,reward,True
            else: 
                return new_state,state,reward,False
        else:
            return new_state,state,reward,False




    
        

if __name__ == '__main__':
    with open('Ourdataset_toy_dataset.pkl','rb') as f:
        dataset = pickle.load(f)
    env  = environment(dataset)
    state_init = env.reset()
    action_space = state_init.get_action_space()
    init_action = get_init_action(action_space)
    new_state,_,done = env.step(state_init,init_action)


    
    
    