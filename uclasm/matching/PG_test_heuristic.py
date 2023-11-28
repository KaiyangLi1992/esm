import pickle
import sys 
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/rlmodel") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/") 
from matching.environment import environment,get_init_action
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from PG_matching import policy_network
from PG_structure import update_state

device = torch.device('cuda:0')
matching_file_name = './data/unEmail_testset_dens_0.2_n_8_num_100_10_05_matching.pkl'   # 获取文件名
with open(matching_file_name,'rb') as f:
    matchings = pickle.load(f)

def calculate_cost(small_graph, big_graph, mapping):
    cost = 0

    # 检查小图的边在大图中是否存在
    for edge in small_graph.edges():
        mapped_edge = (mapping[edge[0]], mapping[edge[1]])
        if not big_graph.has_edge(*mapped_edge):
            cost += 1

    # 检查大图中与小图映射节点相关的边在小图中是否存在
    for small_node in small_graph.nodes():
        mapped_node = mapping[small_node]
        # 遍历与映射节点相邻的大图节点
        for neighbor in big_graph.neighbors(mapped_node):
            # 检查大图的邻居是否在映射中
            if neighbor in mapping.values():
                # 查找原小图中对应的节点
                inv_map = {v: k for k, v in mapping.items()}
                small_neighbor = inv_map[neighbor]
                if not small_graph.has_edge(small_node, small_neighbor):
                    cost += 1

    return cost

def test_checkpoint_model(test_dataset):
    
    # 加载测试环境
    env = environment(test_dataset)
    
    
    total_rewards = []
    steps = []
    first_costs = []
    costs = []
    number_egdes = []
    for episode in range(50):
        
        state_init = env.reset()
        number_egdes.append(state_init.g1.number_of_edges())
        update_state(state_init,env.threshold)
        action = state_init.get_action_heuristic()
        # state_init.action_space = state_init.get_action_space()
        # action = get_init_action(state_init.action_space,state_init.globalcosts)
        new_state, state, reward, done = env.step(state_init, action)
        stack = [state_init]
        solution = []
        

        step = 0 
        while stack:
            state = stack.pop()
            update_state(state,env.threshold)

            if np.any(np.all(state.candidates == False, axis=1)):
                continue
            action = state.get_action_heuristic()
            step += 1

           
            new_state,state, reward, done = env.step(state, action)
            stack.append(state)   
          
            if done:
                costs.append(calculate_cost(new_state.g1,new_state.g2,new_state.nn_mapping))
                print(costs[-1])
                break
            else:
                stack.append(new_state)
   
    return sum(costs)/len(costs)


# 使用该函数测试多个检查点
with open('/home/kli16/ISM_custom/esm_NSUBS/esm/data/unEmail_testset_dens_0.2_n_8_num_100_10_05_RWSE.pkl','rb') as f:
    test_dataset = pickle.load(f)
# checkpoints = [f'/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/matching/checkpoint_{i}.pth' for i in range(18000, 50000, 1000)]
# for checkpoint in checkpoints:
average_cost = test_checkpoint_model(test_dataset)
    # print(checkpoint)
print(average_cost)

