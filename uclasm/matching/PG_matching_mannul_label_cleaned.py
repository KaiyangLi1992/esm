import time
import torch.nn.functional as F
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

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import networkx as nx
from PG_structure import update_state
import random
import gc
import datetime

from PG_matching_ImitationLearning_concat import policy_network
from environment import environment,get_init_action,calculate_cost
import sys
import argparse

 
dataset_file_name = './data/unEmail_toyset_dens_0.2_n_8_num_1_10_05.pkl'   # 获取文件名
matching_file_name = './data/unEmail_toyset_dens_0.2_n_8_num_1_10_05_matching.pkl'   # 获取文件名
gpu_id = 3     # 获取GPU编号
# device = torch.device(f'cuda:{gpu_id}')


timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')



def get_action_space_emb(action_space,x1,x2):
    result_vectors = []

# 遍历 action_space 中的每个元素
    for action in action_space:
        if action == (-1, -1):
            # 如果元素为 (-1, -1)，创建一个长度为106的全0向量
            result_vectors.append(np.zeros(106))
        else:
            # 否则，将 x1 的第a行和 x2 的第b行相加来创建一个新的106列的向量
            a, b = action
            result_vectors.append(x1[a] + x2[b])

    # 将结果向量堆叠起来创建一个 500x106 的矩阵
    result_matrix = np.vstack(result_vectors)
    return result_matrix



def scale_list(input_list, index):
    # 步骤1：设置index外的元素为0
    scaled_list = torch.zeros_like(input_list)
    scaled_list[index] = input_list[index]

    # 步骤2：计算非零元素的和
    sum_of_elements = torch.sum(scaled_list)

    # 步骤3：找到使总和为1的缩放系数
    if sum_of_elements == 0:
        scale_factor = 0  # 避免除以零
    else:
        scale_factor = 1.0 / sum_of_elements

    # 步骤4：缩放列表中的每个元素
    result_list = scaled_list * scale_factor

    return result_list
import random

def update_and_get_position(lst, tup):
    if tup in lst:
        return lst.index(tup),lst  # 返回tuple在list中的位置
    else:
        position = random.randint(0, len(lst)-1)  # 生成随机位置
        lst.insert(position, tup)  # 将tuple插入到随机位置
        return position,lst
    
# class policy_network(nn.Module):
#     def __init__(self):
#         super(policy_network, self).__init__()
#         self.lstm = nn.LSTM(input_size=106, hidden_size=64, batch_first=True)
#         self.linear = nn.Linear(106, 64)
#         self.softmax = nn.Softmax(dim=1)
#         self.hn = None
#         self.cn = None
    
#     def forward(self, state,action,device):
#         init_x1 = state.g1.init_x
#         init_x2 = state.g2.init_x

#         action_space = state.action_space

#         input_matrix = torch.tensor(get_action_space_emb(action_space,init_x1,init_x2),device=device)
#         u,v= action
#         input_vector = torch.tensor(init_x1[u] +init_x2[v],device=device)

#         if self.hn is None:
#             self.hn = torch.zeros(1, 1, 64).to(input_vector.device)
#         if self.cn is None:
#             self.cn = torch.zeros(1, 1, 64).to(input_vector.device)

#         input_vector = input_vector.float()
#         input_matrix = input_matrix.float()
#         self.hn = self.hn.float()
#         self.cn = self.cn.float()
#         # 输入向量经过LSTM得到A向量
#         output, (self.hn, self.cn) = self.lstm(input_vector.unsqueeze(0).unsqueeze(0), (self.hn, self.cn))
#         A_vector = output.squeeze(0)

#         transformed_matrix = self.linear(input_matrix)

#         # 用A向量乘以矩阵B
#         result_matrix = torch.matmul(A_vector, transformed_matrix.transpose(0, 1))

#         # 将结果向量经过Softmax输出
#         output_vector = self.softmax(result_matrix)[-1]

        
#         return output_vector



with open(dataset_file_name,'rb') as f:
    dataset = pickle.load(f)

with open(matching_file_name,'rb') as f:
    matchings = pickle.load(f)


def update_action_exp(state,action):
    gid = state.g1.graph['gid']
    matching = matchings[gid-1]
    action_1 = matching[action[0]]
    action = (action[0],action_1)
    return action


def main():
    device = torch.device(f'cuda:{gpu_id}')
    print(f"Using device: {device}")

    env  = environment(dataset)
    policy = policy_network().to(device) 
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    rewards = []
    writer = SummaryWriter(f'runs/ImitationLearning/{timestamp}')
    checkpoint_interval = 100
    loss_function = nn.CrossEntropyLoss()
    for episode in range(50000):
        rewards = 0
        state_init = env.reset()
        update_state(state_init,env.threshold)
        stack = [state_init]
        policy.hn = None
        policy.cn = None
        action_exp = state_init.get_action_heuristic()
        action_exp = update_action_exp(state_init,action_exp)
        action = action_exp
        new_state,state,reward,_= env.step(state_init,action_exp)
        stack.append(new_state)
        step = 0 
        rewards+=reward
        probs_buffer = []
        probs_exp_buffer = []
        labels = []
        predicts=[]
        while stack:
            state = stack.pop()
            update_state(state,env.threshold)
            # state.candidates = state.get_candidates(env.threshold)

            # 从state_candidates中找出索引不在state_nn_mapping键中的行
            if np.any(np.all(state.candidates == False, axis=1)):
                continue


            state.action_space = state.get_action_space()
            action_exp = state.get_action_heuristic()
            action_exp = update_action_exp(state,action_exp)
            ind,state.action_space = update_and_get_position(state.action_space, action_exp)
            probs_exp = ind
            step += 1
            probs = policy(state,action,device).to(device)
            max_index = torch.argmax(probs)


            pad_length = 500 - probs.size(0)
            probs = F.pad(probs, (0, pad_length), 'constant', 0)
            probs_buffer.append(probs)
            probs_exp_buffer.append(probs_exp)
            newstate, state,reward, done = env.step(state,action_exp)
            rewards+=reward
            stack.append(newstate)

            labels.append(ind)
            predicts.append(max_index.item())   
            if done:
                break
        probs_buffer = torch.stack(probs_buffer)
        probs_exp_buffer = torch.tensor(probs_exp_buffer,dtype=torch.long).to(device)
        loss = loss_function(probs_buffer, probs_exp_buffer)

        ave_acc = sum(1 for x, y in zip(labels, predicts) if x == y)/len(labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('ave_acc', ave_acc, episode)
        if episode%10==0:
            print(ave_acc)


        if episode % checkpoint_interval == 0:
        # 创建一个检查点每隔几个时期
            checkpoint = {
                'epoch': episode,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # ... (其他你想保存的元数据)
            }
            directory_name = f"ckpt_ImitationLearning/{timestamp}/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            torch.save(checkpoint, f'ckpt_ImitationLearning/{timestamp}/checkpoint_{episode}.pth')
    

if __name__ == '__main__':
    main()



