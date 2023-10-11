import time
import torch.nn.functional as F
start_time = time.time()
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
from matching.environment import environment,get_init_action,calculate_cost
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import networkx as nx
from matching.PG_structure import update_state
import random
import gc
import datetime
with open('/home/kli16/ISM_custom/esm/data/Email_trainset_dens_0.2_n_8_num_2000.pkl','rb') as f:
    dataset = pickle.load(f)

g_list = dataset.gs
g0 = g_list[0].nxgraph
matchings = []
for n in range(1,len(g_list)):
    g1 = g_list[n].nxgraph
    matching = {}
    for i in g1.nodes():
        vec = g1.init_x[i,:]
        mat = g0.init_x
        distances = np.linalg.norm(mat - vec, axis=1, ord=1)
        # 找到最小距离的索引
        min_index = np.argmin(distances)
        # print(distances[min_index])
        matching[i] =  min_index
    matchings.append(matching) 

with open('./data/trainset_matching.pkl','wb') as f:
    pickle.dump(matchings,f)


