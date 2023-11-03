import datetime
import os
import torch
import logging

import sys
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS/esm/GraphGPS")
from torch_geometric.data import Data
import graphgps  # noqa, register custom modules
from graphgps.agg_runs import agg_runs
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from torch_geometric.utils import from_networkx

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger
from graphgps.transform.posenc_stats import compute_posenc_stats
from graphgps.network.gps_model import FeatureEncoder
from argparse import Namespace

def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)

def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)

# args = Namespace(
#     cfg_file='/home/kli16/ISM_custom/esm_NSUBS/esm/GraphGPS/configs/GPS/email-GPS+RWSE.yaml',
#     repeat=1,
#     mark_done=False,
#     opts=['wandb.use', 'False']
# )

# print(args)


# set_cfg(cfg)
# load_cfg(cfg, args)
# custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
# dump_cfg(cfg)
# # Set Pytorch environment
# torch.set_num_threads(cfg.num_threads)

# run_id, seed, split_index  = 0,0,0
# # Set configurations for each run
# custom_set_run_dir(cfg, run_id)
# set_printing()
# cfg.dataset.split_index = split_index
# cfg.seed = seed
# cfg.run_id = run_id
# seed_everything(cfg.seed)
# auto_select_device()


# edge_index = torch.tensor([
#     [0, 1, 1, 2],
#     [1, 0, 2, 1]
# ], dtype=torch.long)

# # 每个节点有一个特征
# x = torch.tensor([
#     [1],
#     [2],
#     [3]
# ], dtype=torch.long)

# # 创建 Data 对象
# data = Data(x=x, edge_index=edge_index)
# # pe_types = ['RWSE']
# is_undirected = True


# pe_enabled_list = []
# for key, pecfg in cfg.items():
#     if key.startswith('posenc_') and pecfg.enable:
#         pe_name = key.split('_', 1)[1]
#         pe_enabled_list.append(pe_name)
#         if hasattr(pecfg, 'kernel'):
#             # Generate kernel times if functional snippet is set.
#             if pecfg.kernel.times_func:
#                 pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
#             logging.info(f"Parsed {pe_name} PE kernel times / steps: "
#                             f"{pecfg.kernel.times}")



# data = compute_posenc_stats(data, pe_enabled_list, is_undirected, cfg)
# encoder = FeatureEncoder(1)
# print(encoder)
# data_=encoder(data)





# import pickle
# dataset_file_name = './data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05_RWSE.pkl'   # 获取文件名
# with open(dataset_file_name,'rb') as f:
#     dataset = pickle.load(f)

# for graph in dataset.gs:
#     nx_graph = graph.nxgraph
#     data_of_graph = from_networkx(nx_graph)
#     data_of_graph.x = data_of_graph.type.unsqueeze(1)
#     data_of_graph = compute_posenc_stats(data_of_graph , pe_enabled_list, is_undirected, cfg)
#     data_=encoder(data_of_graph)
    
 


# dot = make_dot(data_, params=dict(list(encoder.named_parameters()) + [('data', data)]))
# dot.view()
'-'
