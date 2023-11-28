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
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/uclasm/") 


from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver,ParameterSaver
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
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_check/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_check/esm/uclasm/") 


from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver,ParameterSaver
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
from PG_structure import update_state
import random
import gc
import datetime

from PG_matching_ImitationLearning_concat import policy_network
from environment import environment,get_init_action,calculate_cost
import sys
import argparse
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

dataset_file_name = './data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05_RWSE.pkl'   # 获取文件名
matching_file_name = './data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05_matching.pkl'   # 获取文件名
# gpu_id = 3     # 获取GPU编号
# device = torch.device(f'cuda:{gpu_id}')
dim = 47
lr = FLAGS.lr


timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# checkpoint_path = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_check/esm/ckpt_imitationlearning/2023-11-10_21-23-11/checkpoint_99000.pth'
# checkpoint = torch.load(checkpoint_path,map_location=torch.device(FLAGS.device))

def _create_model(d_in_raw):
    if FLAGS.matching_order == 'nn':
        if FLAGS.load_model != 'None':
            load_replace_flags(FLAGS.load_model)
            saver.log_new_FLAGS_to_model_info()
            if FLAGS.glsearch:
                model = GLS() # create here since FLAGS have been updated_create_model
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC() # create here since FLAGS have been updated
            ld = torch.load(FLAGS.load_model, map_location=FLAGS.device)
            model.load_state_dict(ld)
            saver.log_info(f'Model loaded from {FLAGS.load_model}')
        else:
            if FLAGS.glsearch:
                model = GLS()
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC()
        saver.log_model_architecture(model, 'model')
        return model.to(FLAGS.device)
    else:
        return None
    

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


def _get_CS(state,g1,g2):
    result = {i: np.where(row)[0].tolist() for i, row in enumerate(state.candidates)}
    return result


def _preprocess_NSUBS(state):
    g1 = state.g1
    g2 = state.g2
    u = state.action_space[0][0]
    v_li = [action[1] for action in state.action_space]
    CS = _get_CS(state,g1,g2)
    nn_map = state.nn_mapping
    candidate_map = {u:v_li}
    return (g1,g2,u,v_li,nn_map,CS,candidate_map)



def main():
    # 使用示例
    saver = ParameterSaver()
    parameters = {
        'pid': os.getpid(),
        'file_path':os.path.abspath(__file__),
        'learning_rate': lr,
        'd_enc' : FLAGS.d_enc,
        'encoder_structure':FLAGS.encoder_structure,
    }
    saver.save(parameters,file_name=f'{timestamp}.log')

    # checkpoint_path = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/ckpt_imitationlearning/2023-11-19_01-22-18/checkpoint_199000.pth'
    # checkpoint_loaded = torch.load(checkpoint_path,map_location=torch.device(FLAGS.device))


    device = torch.device(FLAGS.device)
    print(f"Using device: {device}")
    model = _create_model(dim).to(device)
    # model.train()
    # model.load_state_dict(checkpoint_loaded['model_state_dict'])
    writer = SummaryWriter(f'plt_imitationlearning/{timestamp}')

    env  = environment(dataset)
 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rewards = []

    for episode in range(200000):
        rewards = 0
        state_init = env.reset()
        update_state(state_init,env.threshold)
        stack = [state_init]
        step = 0 
        labels = []
        predicts=[]
        buffer=[]
        while stack:
            state = stack.pop()
            update_state(state,env.threshold)
            if np.any(np.all(state.candidates == False, axis=1)):
                continue
            action_exp = state.get_action_heuristic()
            state.action_space = state.get_action_space(action_exp)
            action_exp = update_action_exp(state,action_exp)
            ind,state.action_space = update_and_get_position(state.action_space, action_exp)
          
            pre_processed = _preprocess_NSUBS(state)
            out_policy, out_value, out_other = \
                model(*pre_processed,
                    True,
                    graph_filter=None, filter_key=None,
                )
            max_index = torch.argmax(out_policy)
            # labels.append(ind)
            predicts.append(max_index.item())  
            step += 1
            pi_true = torch.zeros(len(state.action_space),device=device)
            pi_true[ind] = 1
            buffer.append((out_policy,pi_true))

            newstate, state,reward, done = env.step(state,action_exp)
            rewards+=reward
            stack.append(newstate)

            labels.append(ind) 
            if done:
                model.reset_cache()
                break

        loss_batch = 0
        for out_policy,pi_true in buffer:
            ce_loss = cross_entropy_smooth(out_policy, pi_true)
            loss_batch += ce_loss
        
        ave_acc = sum(1 for x, y in zip(labels, predicts) if x == y)/len(labels)
       
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        
        
        if episode%10==0:
            print(f"episode: {episode} loss: {loss_batch}")
            writer.add_scalar('Loss', loss_batch, episode)
            writer.add_scalar('ACC', ave_acc, episode)


        if episode % 1000 == 0:
        # 创建一个检查点每隔几个时期
            checkpoint = {
                'epoch': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # ... (其他你想保存的元数据)
            }
            directory_name = f"ckpt_imitationlearning/{timestamp}/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            torch.save(checkpoint, f'ckpt_imitationlearning/{timestamp}/checkpoint_{episode}.pth')
    

if __name__ == '__main__':
    main()

