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
from matching.environment import environment,get_init_action,calculate_cost
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from PG_matching import policy_network
from PG_structure import update_state

device = torch.device('cuda:0')

def test_checkpoint_model(test_dataset):
    # 加载模型和优化器
    # checkpoint = torch.load(checkpoint_path)
    # policy = policy_network().to(device)
    # policy.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
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
        state_init.action_space = state_init.get_action_space()
        action = get_init_action(state_init.action_space,state_init.globalcosts)
        new_state, state, reward, done = env.step(state_init, action)
        stack = [state_init]
        solution = []
        
        # rewards = []
        step = 0 
        while stack:
            state = stack.pop()
            update_state(state,env.threshold)

            if np.any(np.all(state.candidates == False, axis=1)):
                continue
            state.action_space = state.get_action_heuristic()
            step += 1

            action = state.action_space
            new_state,state, reward, done = env.step(state, action)
            stack.append(state)   
          
            if done:
                # print(new_state.globalcosts.min())
                # solution.append(new_state)
                costs.append(calculate_cost(new_state.g1,new_state.g2,new_state.nn_mapping))
                print(costs[-1])
                break
            else:
                stack.append(new_state)
                
        # steps.append(step)
        # first_costs.append(costs[0])

        

    # 返回总奖励，以便在后续可能使用
   
    return sum(costs)/len(costs)


# 使用该函数测试多个检查点
with open('/home/kli16/ISM_custom/esm_NSUBS/esm/data/unEmail_testset_dens_0.2_n_8_num_50_noise_5_10_18.pkl','rb') as f:
    test_dataset = pickle.load(f)
# checkpoints = [f'/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/matching/checkpoint_{i}.pth' for i in range(18000, 50000, 1000)]
# for checkpoint in checkpoints:
average_cost = test_checkpoint_model(test_dataset)
    # print(checkpoint)
print(average_cost)

