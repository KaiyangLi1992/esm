from NSUBS.model.OurSGM.solve_parent_dir import solve_parent_dir
from NSUBS.src.utils import get_user, get_host
import argparse
import torch
import json
# constants
AREA, POSITIVE_PAIRS, COVERAGE, BINARY, EXP_DEPTH, EXP_DEPTH_RAW, DEPTH = \
    'area', 'positive_pairs', 'coverage', 'binary', 'exp_depth', 'exp_depth_raw', 'depth'
ADD_GND_TRUTH_AT_START, ADD_GND_TRUTH_AT_END = 'add_gnd_truth_at_start', 'add_gnd_truth_at_end'
INDUCED, ISOMORPHIC, HOMOMORPHIC = 'induced', 'isomorphic', 'homomorphic'

solve_parent_dir()
parser = argparse.ArgumentParser()

debug = False




main_func = 'here'
parser.add_argument('--main_func', type=str, default=main_func)

parser.add_argument('--filter', type=str, default='DPiso')
parser.add_argument('--order', type=str, default='GQL')
parser.add_argument('--engine', type=str, default='LFTJ')

model = 'our'
parser.add_argument('--model', type=str, default=model)

do_train = True
# do_train = False
parser.add_argument('--do_train', type=bool, default=do_train)

# do_validation = True
do_validation = False
parser.add_argument('--do_validation', type=bool, default=do_validation)

# do_test = True
do_test = False
parser.add_argument('--do_test', type=bool, default=do_test)

num_cs_refinements = 3
parser.add_argument('--num_cs_refinements', type=int, default=num_cs_refinements)

search_constraints = ISOMORPHIC # INDUCED, ISOMORPHIC, HOMOMORPHIC # @@@@@@@@@@@@@@
parser.add_argument('--search_constraints', default=search_constraints)

thresh_throwaway_qsa = 1e-8
parser.add_argument('--thresh_throwaway_qsa', type=float, default=thresh_throwaway_qsa)

reward_method = EXP_DEPTH#EXP_DEPTH#DEPTH# BINARY if loss_type == 'xentropy' else COVERAGE  # TODO
parser.add_argument('--reward_method', type=str, default=reward_method)
if reward_method == EXP_DEPTH:
    reward_method_exp_coeff = 100
    reward_method_normalize = True
    parser.add_argument('--reward_method_exp_coeff', type=str, default=reward_method_exp_coeff)
    parser.add_argument('--reward_method_normalize', type=bool, default=reward_method_normalize)
    cpuct_multiplier = reward_method_exp_coeff if not reward_method_normalize else 1.0
else:
    cpuct_multiplier = 1.0
if do_train:
    debug_embeddings_every_k_iters = None
    parser.add_argument('--debug_embeddings_every_k_iters', type=int,
                        default=debug_embeddings_every_k_iters)
    loss_type = 'mse_bounded'#'mse_bounded'#'pos_neg_sampling'
    parser.add_argument('--loss_type', type=str, default=loss_type)
    regularization = 0.05  # TODO -> pos neg sampling + softmax
    parser.add_argument('--regularization', type=float, default=regularization)
    add_gnd_truth = ADD_GND_TRUTH_AT_START # None  # ADD_GND_TRUTH_AT_START  # ADD_GND_TRUTH_AT_END
    parser.add_argument('--add_gnd_truth', type=str, default=add_gnd_truth)
    parser.add_argument('--mark_best_as_true', type=bool, default=True)
    eps_decay_config = {
        'start_eps': 0.0,
        # 'start_eps': 0.5, # TODO: for debugging to see NN calls
        # 'start_eps': 0.0,
        'end_eps': 0.0,
        'num_iters': 5
    }
    parser.add_argument('--eps_decay_config', default=eps_decay_config)


    train_dataset = 'imsm-dblp_unlabeled'

    sample_size = [5000,10000,50000] if 'patents' in train_dataset or 'eu2005' in train_dataset else [None]
    parser.add_argument('--train_dataset', type=str, default=train_dataset)
    parser.add_argument('--val_dataset', type=str, default=train_dataset)
    parser.add_argument('--train_sample_size_li', type=list, default=sample_size)
    parser.add_argument('--val_sample_size_li', type=list, default=sample_size)

    assert 'imsm' in train_dataset


    subgroup = 'list'
    if subgroup == 'list':
        if debug:
            train_subgroup_list = ['dense_64']
            train_path_indices = [[95,96]]
        else:
            train_subgroup_list = ['dense_64']
            train_path_indices = \
                [[0,5]] * (4 if 'imsm-wordnet' not in train_dataset and 'imsm-human' not in train_dataset else 4) + \
                [[0,50]] * 3
    else:
        train_subgroup_list = None
        train_path_indices = [0,5]

    parser.add_argument('--train_subgroup', type=str, default=subgroup)
    parser.add_argument('--train_subgroup_list', type=list, default=train_subgroup_list)
    parser.add_argument('--train_path_indices', default=train_path_indices)

    parser.add_argument('--val_subgroup', type=str, default='dense_64')
    parser.add_argument('--val_subgroup_list', type=list, default=None)
    parser.add_argument('--val_path_indices', default=[95,100])

    
    # pretrain = True
    pretrain = False  # now regularization
    parser.add_argument('--pretrain', type=bool, default=pretrain)

    if pretrain:
        pretrain_iters = 100000
        parser.add_argument('--pretrain_iters', type=int, default=pretrain_iters)

    # imitation = True
    imitation = False  # TODO
    parser.add_argument('--imitation', type=bool, default=imitation)

    if imitation:
        imitation_iters = 200
        parser.add_argument('--imitation_iters', type=int, default=imitation_iters)

    # num_eps = 1
    # parser.add_argument('--num_eps', type=int, default=num_eps)

    show_precision = False
    parser.add_argument('--show_precision', type=bool, default=show_precision)

    buffer_size = 128
    #buffer_size = 16
    parser.add_argument('--buffer_size', type=int, default=buffer_size)

    batch_size = 1
    parser.add_argument('--batch_size', type=int, default=batch_size)

    parser.add_argument('--print_buffer_stats', type=bool, default=True)

    # prune_trivial_rewards = True
    prune_trivial_rewards = False # for debugging
    parser.add_argument('--prune_trivial_rewards', type=bool, default=prune_trivial_rewards)

    parser.add_argument('--print_rpreds_stats', type=bool, default=True)

    parser.add_argument('--glsearch', type=bool, default=False)
    parser.add_argument('--num_epochs_per_learn', type=int, default=128)
    parser.add_argument('--num_outer_epoch', type=int, default=64)

    skip_data_leakage = False
    parser.add_argument('--skip_data_leakage', type=bool, default=skip_data_leakage)

if do_test:
    test_dataset = 'imsm-dblp_unlabeled'
    parser.add_argument('--test_dataset', type=str, default=test_dataset)

    if 'imsm' in test_dataset:
        test_subgroup = 'dense_32'
        test_subgroup = 'dense_64'
        assert 'li' not in test_subgroup
        parser.add_argument('--test_subgroup', type=str, default=test_subgroup)
        parser.add_argument('--test_subgroup_list', type=list, default=None)
        parser.add_argument('--test_path_indices', default=[100,200])

    if test_dataset == 'mix':
        mix = [
            ('ZINC', 5, 16),  # 38 nodes

        ]
        parser.add_argument('--mix', type=list, default=mix)
load_model = 'None'
parser.add_argument('--load_model', type=str, default=load_model)

append_ldf = True
parser.add_argument('--append_ldf', type=str, default=append_ldf)

if do_train:
    parser.add_argument('--val_every_games', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1000)

if do_train or do_validation:
    matching_order = 'nn'
else:
    assert do_test
    if load_model == 'None':
        # matching_order = 'DAF'  # random v selection
        matching_order = 'GraphQL' # random v selection
        # matching_order = 'nn' # learned v selection
        # matching_order = 'mcsp'
    else:
        matching_order = 'nn'
parser.add_argument('--matching_order', type=str, default=matching_order)
parser.add_argument('--use_is_early_pruning', type=bool, default=False) # @@@@@@@@@@@@@@

MCTS_train = True
MCTS_test = False
parser.add_argument('--MCTS_train', type=bool, default=MCTS_train)
parser.add_argument('--MCTS_test', type=bool, default=MCTS_test)
parser.add_argument('--MCTS_printQU', type=bool, default=True)

if MCTS_train or MCTS_test:
    MCTS_num_iters_max = 25 if debug else 120
    parser.add_argument('--MCTS_num_iters_max', type=int, default=MCTS_num_iters_max)
    MCTS_num_iters_per_action = 2.0 if debug else 10.0
    parser.add_argument('--MCTS_num_iters_per_action', type=float, default=MCTS_num_iters_per_action)

    MCTS_temp = 5.0
    parser.add_argument('--MCTS_temp', type=float, default=MCTS_temp)

    MCTS_temp_inner = 1.0
    parser.add_argument('--MCTS_temp_inner', type=float, default=MCTS_temp_inner)

    # MCTS_cpuct = 1
    MCTS_cpuct = 10
    parser.add_argument('--MCTS_cpuct', type=float, default=cpuct_multiplier*MCTS_cpuct)

    MCTS_eps_in_U = 1e-8
    parser.add_argument('--MCTS_eps_in_U', type=float, default=MCTS_eps_in_U)

    MCTS_backup_to_real_root = True
    parser.add_argument('--MCTS_backup_to_real_root', type=bool, default=MCTS_backup_to_real_root)

'''
OurGMNv1: our_train_imsm-youtube_unlabeled_list_nn_300_2022-05-14T20-00-58.497315
OurGMNv2 our encoder: our_train_imsm-youtube_unlabeled_list_nn_300_2022-05-14T20-00-50.199613
'''
################################################
if matching_order == 'nn':
    d_enc = 128
    parser.add_argument('--d_enc', type=int, default=d_enc)
    parser.add_argument('--encoder_type', type=str, default='mlp')
    use_NN_for_u_score = False  # use NN for u score instead of DAF
    parser.add_argument('--use_NN_for_u_score', type=bool, default=use_NN_for_u_score)

    parser.add_argument('--dvn_config', default=None)
    # cache_target_embeddings = dvn_config['preencoder']['type'] == 'mlp' and not dvn_config['encoder']['q2t']
    parser.add_argument('--cache_embeddings', type=bool, default=True)#cache_target_embeddings)
    parser.add_argument('--k_sample_cross_graph', type=int, default=None)#cache_target_embeddings)
else:
    use_NN_for_u_score = False  # use NN for u score instead of DAF
    parser.add_argument('--use_NN_for_u_score', type=bool, default=use_NN_for_u_score)

if do_train:
    regret_iters_train = 1
    parser.add_argument('--regret_iters_train', type=int, default=regret_iters_train)

regret_iters_test = 1
parser.add_argument('--regret_iters_test', type=int, default=regret_iters_test)

use_node_mask_diameter = False
parser.add_argument('--use_node_mask_diameter', type=bool, default=use_node_mask_diameter)




num_iters_threshold = -1
parser.add_argument('--num_iters_threshold', type=int, default=num_iters_threshold)


timeout = 60 if debug else 300 # 2 minutes # @@@@@@@@@@
parser.add_argument('--timeout', type=int, default=timeout)

time = '2023-10-20_11-44-50'
parser.add_argument('--time', type=str, default=time)


parser.add_argument('--time_analysis', type=bool, default=debug)

if do_train:
    learning_timeout = 120#180  # 3 minutes
    parser.add_argument('--learning_timeout', type=int, default=learning_timeout)
    timeout_val = 120 # shouldn't need to hit the timeout here!
    num_iters_threshold_val = 200
    parser.add_argument('--timeout_val', type=int, default=timeout_val)
    parser.add_argument('--num_iters_threshold_val', type=int, default=num_iters_threshold_val)


plot_tree = debug
parser.add_argument('--plot_tree', type=bool, default=plot_tree)

# plot_solution = True
plot_solution = False
parser.add_argument('--plot_solution', type=bool, default=plot_solution)

# plot_logs = True
plot_logs = False
parser.add_argument('--plot_logs', type=bool, default=plot_logs)

# save_search = True
save_search = False
parser.add_argument('--save_search', type=bool, default=save_search)

gpu = 4
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')

# device = 'cpu'
parser.add_argument('--device', default=device)


# 'encoder1', 'encoder2', 'encoder3','encoder4' #
encoder_structure = 'encoder2'
parser.add_argument('--encoder_structure', default=encoder_structure)

graphgps_config_path = 'email_GateGCN_RSWE.yaml'
parser.add_argument('--graphgps_config_path', default=graphgps_config_path)

lr= 1e-4
parser.add_argument('--lr',type=float, default=lr)


"""
Other info.
"""

fix_randomness = True  # TODO: may have issue with result
# fix_randomness = False
parser.add_argument('--fix_randomness', type=bool, default=fix_randomness)

if fix_randomness:
    parser.add_argument('--random_seed', type=bool, default=123)

parser.add_argument('--skip_if_action_space_less_than', type=int, default=None) # TODO: maybe try it during testing

parser.add_argument('--apply_norm', type=bool, default=True)

# FLAGS.ckpt


parser.add_argument('--ckpt', type=str, default='')

parser.add_argument('--user', default=get_user())
parser.add_argument('--hostname', default=get_host())
FLAGS = parser.parse_args()

FLAGS.dvn_config = {
    'preencoder': {
        'type': 'concat+mlp'
    },
    'encoder': {
        'type': 'GNNConsensusEncoder',
        'gnn_type': 'OurGMNv2',     
        'gnn_subtype': 'graphgps',
        # 'hidden_gnn_dims': [FLAGS.d_enc, FLAGS.d_enc], 
        'hidden_gnn_dims': [FLAGS.d_enc, FLAGS.d_enc, FLAGS.d_enc, FLAGS.d_enc],  # 5. 更新
        'shared_gnn_weights': False,
        'shared_encoder': False,
        'q2t': True,
        't2q': True,
        'consensus_cfg_li': None
    },
    'decoder_dvn': {
        'type': 'Query',
        'simple_decoder': {
            'mlp_att': [FLAGS.d_enc, 8, 1],  # 5. 更新
            'mlp_val': [FLAGS.d_enc, FLAGS.d_enc],  # 5. 更新
            'mlp_final': [FLAGS.d_enc, 32, 16, 8, 4, 1]  # 5. 更新
        },
    },
    'decoder_policy': {
        'type': 'bilinear_custom',
        'similarity_decoder': {
            'mlp_in_dims': [FLAGS.d_enc, FLAGS.d_enc],  # 5. 更新
            'mlp_out_dims': [2*FLAGS.d_enc, 32, 16, 8, 1],
            'g_emb': FLAGS.d_enc,  # 5. 更新
        }
    }
}
