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
from PG_matching_imitationlearning import update_and_get_position#,
from PG_structure import update_state
from PG_matching_mannul_label import update_action_exp

device = torch.device('cuda:0')
from PG_matching_ImitationLearning_concat import policy_network
# from PG_matching_imitationlearning import  policy_network

import os
import shutil

with open('./data/Email_testset_dens_0.2_n_8_num_50_10_05_matching.pkl','rb') as f:
    matchings = pickle.load(f)
def update_action_exp(state,action):
    gid = state.g1.graph['gid']
    matching = matchings[gid-1]
    action_1 = matching[action[0]]
    action = (action[0],action_1)
    return action

def clear_directory(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录
        except Exception as e:
            print(f"删除 {file_path} 失败。原因: {e}")

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
    labels = []
    predicts = []
    for episode in range(50):
        state_init = env.reset()
        update_state(state_init,env.threshold)
        state_init.action_space = state_init.get_action_space()
        action = get_init_action(state_init.action_space,state_init.globalcosts)
        action_exp = update_action_exp(state_init,action)
        new_state, state, reward, done = env.step(state_init, action_exp)
        stack = [state_init]
        solution = []
        costs = []
        rewards = []
        # step = 0 
        while stack:
            state = stack.pop()
            update_state(state,env.threshold)

            if np.any(np.all(state.candidates == False, axis=1)):
                continue
            state.action_space = state.get_action_space()
            # step += 1
            action_exp = update_action_exp(state,state.action_space[0])
            ind, state.action_space =  update_and_get_position(state.action_space,action_exp)
            probs = policy(state,action,device).to(device)
            m = Categorical(probs)
            action_index = m.sample()
            action = state.action_space[action_index]
            action_exp = state.action_space[ind]
            new_state,state, reward, done = env.step(state, action_exp)
            max_index = torch.argmax(probs)

            labels.append(ind)
            predicts.append(max_index.item())
            
            # rewards.append(reward)
            stack.append(new_state)   
          
            if done:
                # print(new_state.globalcosts.min())
                # solution.append(new_state)
                costs.append(new_state.globalcosts.min())
                break
            else:
                stack.append(new_state)
                
        # print(f'Step:{step}') 
        # steps.append(step)
        first_costs.append(costs[0])

        

    # 返回总奖励，以便在后续可能使用
    # return sum(first_costs)/len(first_costs)
    return sum(1 for x, y in zip(labels, predicts) if x == y)/len(labels)
    # return sum(steps)/len(steps)

time = '2023-10-05_12-59-10'
# 使用该函数测试多个检查点
with open('./data/Email_testset_dens_0.2_n_8_num_50_10_05_new.pkl','rb') as f:
    test_dataset = pickle.load(f)
# checkpoints = [f'/home/kli16/ISM_custom/esm/ckpt_ImitationLearning/{time}/checkpoint_{i}.pth' for i in
#  range(18000, 50000, 1000)]
try:
    clear_directory(f'/home/kli16/ISM_custom/esm/runs_test_acc/{time}/')
except:
    pass
writer = SummaryWriter(f'/home/kli16/ISM_custom/esm/runs_test_acc/{time}/')
for i in range(0, 1100, 100):
    checkpoint = f'/home/kli16/ISM_custom/esm/ckpt_ImitationLearning/{time}/checkpoint_{i}.pth'
    average_acc = test_checkpoint_model(checkpoint, test_dataset)
    writer.add_scalar('Metrics/Cost', average_acc, i)
    print(checkpoint)
    print(average_acc)

