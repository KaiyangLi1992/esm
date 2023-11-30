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
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_reorder/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_reorder/esm/uclasm/") 

from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver
from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.train import train
from NSUBS.model.OurSGM.test import test
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.src.utils import OurTimer, save_pickle
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.model.OurSGM.train import cross_entropy_smooth



import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import networkx as nx
# from PG_structure import update_state
import random
import gc
import datetime

from environment import environment,get_init_action,calculate_cost
import sys
from PG_matching_RL_undirect import _create_model,_get_CS,_preprocess_NSUBS,update_and_get_position
matching_file_name = './data/unEmail_testset_dens_0.2_n_8_num_100_10_05_matching.pkl'   # 获取文件名
with open(matching_file_name,'rb') as f:
    matchings = pickle.load(f)

def update_action_exp(state,action):
    gid = state.g1.graph['gid']
    matching = matchings[gid-1]
    action_1 = matching[action[0]]
    action = (action[0],action_1)
    return action

device = FLAGS.device
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
    
model = _create_model(47)
def test_checkpoint_model(ckpt_pth,test_dataset):
    
    checkpoint = torch.load(ckpt_pth,map_location=torch.device(FLAGS.device))
    model.load_state_dict(checkpoint['model_state_dict'])

    env = environment(test_dataset)
    # total_rewards = []
    # steps = []
    # first_costs = []
    # costs = []
    costs = []
    for episode in range(50):
        state_init = env.reset()
    
        stack = [state_init]
        


        while stack:
            state = stack.pop()
            # update_state(state,env.threshold)

            if np.any(np.all(state.candidates == False, axis=1)):
                continue
            
            # action_exp = state.get_action_heuristic()
            state.action_space = state.get_action_space(env.order)
            # action_exp = update_action_exp(state,action_exp)
            # ind,state.action_space = update_and_get_position(state.action_space, action_exp)
            pre_processed = _preprocess_NSUBS(state)
            out_policy, out_value, out_other = \
                model(*pre_processed,
                    True,
                    graph_filter=None, filter_key=None,
                )
            action_prob = F.softmax(out_policy - out_policy.max()) + 1e-10

            m = Categorical(action_prob)
            action_ind = m.sample()
            action = state.action_space[action_ind]
            newstate, state,reward, done = env.step(state,action)
            
            stack.append(newstate)

            # predicts.append(max_index.item())   
            if done:
                costs.append(calculate_cost(newstate.g1,newstate.g2,newstate.nn_mapping))
                model.reset_cache()
                break


            # step += 1

            
        # steps.append(step)
        # first_costs.append(costs[0])

        

    # 返回总奖励，以便在后续可能使用
    # print(sum(costs)/len(costs))
    return sum(costs)/len(costs)


# 使用该函数测试多个检查点
with open('/home/kli16/ISM_custom/esm_NSUBS/esm/data/unEmail_testset_dens_0.2_n_8_num_100_10_05_RWSE.pkl','rb') as f:
    test_dataset = pickle.load(f)
# checkpoints = [f'/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/matching/checkpoint_{i}.pth' for i in range(0, 90000, 500)]
# for ckpt_pth in checkpoints:
#     average_cost = test_checkpoint_model(test_dataset,ckpt_pth)
    # print(checkpoint)
# print(average_cost)
# time = FLAGS.time
time = '2023-11-25_23-16-54'
try:
    clear_directory(f'/home/kli16/ISM_custom/esm_NSUBS_RWSE_reorder/esm/runs_RL_test/{time}/')
except:
    pass
writer = SummaryWriter(f'/home/kli16/ISM_custom/esm_NSUBS_RWSE_reoder/esm/runs_RL_test/{time}/')
for i in range(0, 120000, 500):
    checkpoint = f'/home/kli16/ISM_custom/esm_NSUBS_RWSE_reorder/esm/ckpt_RL/{time}/checkpoint_{i}.pth'
    average_cost = test_checkpoint_model(checkpoint, test_dataset)
    writer.add_scalar('Metrics/Cost', average_cost, i)
    print(checkpoint)
    print(average_cost)

