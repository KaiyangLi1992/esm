import pickle
import sys 
import random
import torch
from laptools import clap
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
sys.path.append("/home/kli16/ISM_custom/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm/rlmodel") 
sys.path.append("/home/kli16/ISM_custom/esm/uclasm/") 
from matching.environment import environment,get_init_action
from matching.PG_structure import State
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import os
import shutil
from collections import defaultdict
from matching.matching_utils import inspect_channels, MonotoneArray




# def generate_random_digraph(num_nodes):
#     G = nx.DiGraph()
#     attr_dict = defaultdict(set)

#     for i in range(num_nodes):
#         type_value = random.randint(1, 2)
#         G.add_node(i, type=type_value)
#         attr_dict[type_value].add(i)
    
#     for i in range(num_nodes):
#         G.add_edge(i, (i+1) % num_nodes)

#     G.attr_dict = attr_dict
    
#     return G



# g1 = generate_random_digraph(2)
# g2 = generate_random_digraph(3)

# with open('g1.pkl','wb') as f:
#     pickle.dump(g1, f)


# with open('g2.pkl','wb') as f:
#     pickle.dump(g2, f)


# 创建两个DiGraph实例




# with open('g1.pkl', 'rb') as f:
#     # 使用pickle.load() 从文件中加载对象
#     g1 = pickle.load(f)

# with open('g2.pkl', 'rb') as f:
#     # 使用pickle.load() 从文件中加载对象
#     g2 = pickle.load(f)

g1 = nx.DiGraph()

# 添加节点
nodes = [0, 1, 2]
g1.add_nodes_from(nodes)

# 为每个节点设置 'type' 属性为 1
for node in nodes:
    g1.nodes[node]['type'] = 1

# 添加边
edges = [(0, 1), (1, 0), (0, 2)]
g1.add_edges_from(edges)


g2= nx.DiGraph()

# 添加节点
nodes = [0, 1, 2,3]
g2.add_nodes_from(nodes)

# 为每个节点设置 'type' 属性为 1
for node in nodes:
    g2.nodes[node]['type'] = 1

# 添加边
edges = [(0, 1), (0, 3), (1, 2),(2,1),(3,2),(1,0)]
g2.add_edges_from(edges)





g1.graph['gid'] =1 
g2.graph['gid'] =2

nn_mapping = {0:0}



state = State(g1,g2,nn_mapping=nn_mapping)
# state.candidates = state.get_candidates(np.inf) 

changed_cands = np.ones((len(state.g1.nodes),), dtype=np.bool)

old_candidates = state.candidates.copy()
state.localcosts[:] = np.maximum(state.localcosts, state.get_localcosts())
state.globalcosts[:] = np.maximum(state.globalcosts, state.get_globalcosts())

# TODO: Does this break if nodewise changes the candidates?
while True:
    
    state.localcosts[:] = np.maximum(state.localcosts, state.get_localcosts())
    state.globalcosts[:] = np.maximum(state.globalcosts, state.get_globalcosts())
    state.candidates = state.get_candidates()
   
    changed_cands = np.any(state.candidates != old_candidates, axis=1)
    if ~np.any(changed_cands):
        break
    
    old_candidates = state.candidates.copy()

_=1




    




