import pickle
import sys 
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
sys.path.append("/home/kli16/ISM_custom/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm/rlmodel") 
sys.path.append("/home/kli16/ISM_custom/esm/uclasm/") 
from matching.environment import environment,get_init_action
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from PG_matching import policy_network
from PG_structure import update_state

device = torch.device('cuda:0')

def test_checkpoint_model(checkpoint_path, test_dataset):
    # 加载模型和优化器
    checkpoint = torch.load(checkpoint_path)
    policy = policy_network().to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载测试环境
    env = environment(test_dataset)
    
    
    total_rewards = []
    steps = []
    first_costs = []
    for episode in range(50):
        state_init = env.reset()
        update_state(state_init,env.threshold)
        state_init.action_space = state_init.get_action_space()
        action = get_init_action(state_init.action_space,state_init.globalcosts)
        new_state, state, reward, done = env.step(state_init, action)
        stack = [state_init]
        solution = []
        costs = []
        # rewards = []
        step = 0 
        while stack:
            state = stack.pop()
            update_state(state,env.threshold)

            if np.any(np.all(state.candidates == False, axis=1)):
                continue
            state.action_space = state.get_action_heuristic()
            step += 1
            # probs = policy(state, new_state,action,device)
            # m = Categorical(probs)
            # action_index = m.sample()
            # action = state.action_space[action_index]
            action = state.action_space
            new_state,state, reward, done = env.step(state, action)
            # rewards.append(reward)
            stack.append(state)   
          
            if done:
                print(new_state.globalcosts.min())
                solution.append(new_state)
                costs.append(new_state.globalcosts.min())
                break
            else:
                stack.append(new_state)
                
        # print(f'Step:{step}') 
        steps.append(step)
        first_costs.append(costs[0])

        

    # 返回总奖励，以便在后续可能使用
    print(sum(first_costs)/len(first_costs))
    return sum(steps)/len(steps)


# 使用该函数测试多个检查点
with open('/home/kli16/ISM_custom/esm_only_rl/esm/Email_testset_dens_0.2_n_8_num_50_same_embedding.pkl','rb') as f:
    test_dataset = pickle.load(f)
checkpoints = [f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/checkpoint_{i}.pth' for i in range(18000, 50000, 1000)]
for checkpoint in checkpoints:
    average_step = test_checkpoint_model(checkpoint, test_dataset)
    print(checkpoint)
    print(average_step)

