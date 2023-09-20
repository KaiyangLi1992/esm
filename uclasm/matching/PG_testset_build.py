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
with open('Email_trainset_dens_0.5_n_10.pkl','rb') as f:
    train_dataset = pickle.load(f)
with open('Email_testset_dens_0.5_n_10.pkl','rb') as f:
    test_dataset = pickle.load(f)
test_dataset.gs[0].nxgraph.init_x = train_dataset.gs[0].nxgraph.init_x

with open('Email_testset_dens_0.5_n_10_new.pkl','wb') as f:
    pickle.dump(test_dataset,f)