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
    for episode in range(40):
        state_init = env.reset()
        action_space = state_init.action_space
        action = get_init_action(action_space)
        state, reward, done = env.step(state_init, action)
        
        rewards = []
        for t in range(1, 10000): # 这里的范围可能需要调整
            probs = policy(state, action,device)
            m = Categorical(probs)
            action_index = m.sample()
            action = state.action_space[action_index]
            state, reward, done = env.step(state, action)
            rewards.append(reward)
            if done:
                break

        total_reward = sum(rewards)
        total_rewards.append(total_reward)
    Average_reward  = sum(total_rewards)/len(total_rewards)
    print(f'Total reward for checkpoint {checkpoint_path}: {Average_reward}')

    # 返回总奖励，以便在后续可能使用
    return total_rewards


# 使用该函数测试多个检查点
with open('Email_testset_dens_0.5_n_10_new.pkl','rb') as f:
    test_dataset = pickle.load(f)
checkpoints = [f'checkpoint_{i}.pth' for i in range(0, 50000, 1000)]
for checkpoint in checkpoints:
    test_checkpoint_model(checkpoint, test_dataset)
