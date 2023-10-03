import pickle
import sys 
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
sys.path.append("/home/kli16/ISM_custom/esm_only_rl/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_only_rl/esm/rlmodel") 
sys.path.append("/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/") 
from matching.environment import environment,get_init_action,calculate_cost
from matching.environment import environment,get_init_action,calculate_cost
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from PG_matching import policy_network

device = torch.device('cuda:1')
def get_last_two_dirs(path):
    first_dir = os.path.basename(path)
    second_dir = os.path.basename(os.path.dirname(path))
    return second_dir, first_dir

def find_all_folders(path):
    folder_list = []
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            folder_list.append(os.path.join(root, dir_name))
    return folder_list

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
    costs = []
    for episode in range(50):
        state_init = env.reset()
        action_space = state_init.action_space
        action = get_init_action(action_space)
        rewards = []
        state, reward, done = env.step(state_init, action)
        rewards.append(reward)
        
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
        cost = calculate_cost(state.g1,state.g2,state.nn_mapping)
        costs.append(cost)
    #     print(total_reward)
    #     print(len(state.g1.edges()))
    #     print(state.g1.graph['gid'])
    ave_reward  = sum(total_rewards)/len(total_rewards)
    ave_cost = sum(costs)/len(costs)
    # print(f'Total reward for checkpoint {checkpoint_path}: {Average_reward}')

    # 返回总奖励，以便在后续可能使用
    return ave_reward,ave_cost 


# 使用该函数测试多个检查点
with open('/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/Email_testset_dens_0.2_n_8_num_50_new.pkl','rb') as f:
# with open('/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/Email_trainset_dens_0.2_n_8_num_2000.pkl','rb') as f:
    test_dataset = pickle.load(f)
model = 'ImitationLearning'
path = f'/home/kli16/ISM_custom/esm/uclasm/matching/ckpt_{model}'
folders = find_all_folders(path)

# for folder in folders:
#     print(folder)

# checkpoints = [f'{folders[0]}/checkpoint_{i}.pth' for i in range(0, 50000, 1000)]
# for checkpoint in checkpoints:
#     test_checkpoint_model(checkpoint, test_dataset)


for folder in folders:
    dir1, dir2 = get_last_two_dirs(folder)
    writer = SummaryWriter(f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/runs_test/{dir1}/{dir2}/')
    for i in  range(0, 50000, 100):
        try:
            folder = f'/home/kli16/ISM_custom/esm/uclasm/matching/{dir1}/{dir2}/'
            checkpoint = f'{folder}/checkpoint_{i}.pth'
            ave_reward,ave_cost  = test_checkpoint_model(checkpoint, test_dataset)
            # 记录损失值
            writer.add_scalar('Metrics/Cost', ave_cost, i)

            # 记录奖励值
            writer.add_scalar('Metrics/Reward', ave_reward, i)
        except:
            pass 



model = 'smooth'
path = f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/ckpt_{model}_0927'
folders = find_all_folders(path)

# for folder in folders:
#     print(folder)

# checkpoints = [f'{folders[0]}/checkpoint_{i}.pth' for i in range(0, 50000, 1000)]
# for checkpoint in checkpoints:
#     test_checkpoint_model(checkpoint, test_dataset)


for folder in folders:
    dir1, dir2 = get_last_two_dirs(folder)
    writer = SummaryWriter(f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/runs_test/{dir1}/{dir2}/')
    for i in  range(0, 50000, 1000):
        try:
            folder = f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/ckpt_{model}_0927/{dir2}/'
            checkpoint = f'{folder}/checkpoint_{i}.pth'
            ave_reward,ave_cost  = test_checkpoint_model(checkpoint, test_dataset)
            # 记录损失值
            writer.add_scalar('Metrics/Cost', ave_cost, i)

            # 记录奖励值
            writer.add_scalar('Metrics/Reward', ave_reward, i)
        except:
            pass 

model = 'original'
path = f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/ckpt_{model}_0927'
folders = find_all_folders(path)

# for folder in folders:
#     print(folder)

# checkpoints = [f'{folders[0]}/checkpoint_{i}.pth' for i in range(0, 50000, 1000)]
# for checkpoint in checkpoints:
#     test_checkpoint_model(checkpoint, test_dataset)


for folder in folders:
    dir1, dir2 = get_last_two_dirs(folder)
    writer = SummaryWriter(f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/runs_test/{dir1}/{dir2}/')
    for i in  range(0, 50000, 1000):
        try:
            folder = f'/home/kli16/ISM_custom/esm_only_rl/esm/uclasm/matching/ckpt_{model}_0927/{dir2}/'
            checkpoint = f'{folder}/checkpoint_{i}.pth'
            ave_reward,ave_cost  = test_checkpoint_model(checkpoint, test_dataset)
            # 记录损失值
            writer.add_scalar('Metrics/Cost', ave_cost, i)

            # 记录奖励值
            writer.add_scalar('Metrics/Reward', ave_reward, i)
        except:
            pass 


