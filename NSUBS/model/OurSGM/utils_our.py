from NSUBS.model.OurSGM.solve_parent_dir import cur_folder
from NSUBS.src.utils import sorted_nicely, get_ts, load
from config import FLAGS
from os.path import join, dirname
import torch
from collections import OrderedDict
from scipy.stats import mstats
import numpy as np
import matplotlib.pyplot as plt
# try:
import seaborn as sns

from NSUBS.model.OurSGM.config import FLAGS, AREA, POSITIVE_PAIRS, COVERAGE, BINARY, DEPTH, EXP_DEPTH, EXP_DEPTH_RAW


def get_reward_terminal(nn_map_len, nn_map_len_max, is_terminal):
    if is_terminal:
        if FLAGS.reward_method == DEPTH:
            reward_terminal = 0.0
        elif FLAGS.reward_method in EXP_DEPTH:
            reward_terminal = 0.0
        elif FLAGS.reward_method == BINARY:
            reward_terminal = (np.exp(np.log(FLAGS.reward_method_exp_coeff + 1) * nn_map_len / nn_map_len_max) - 1)
            if FLAGS.reward_method_normalize:
                reward_terminal /= FLAGS.reward_method_exp_coeff
                assert reward_terminal <= 1.0, print(f'reward terminal: {reward_terminal}')
        else:
            assert False
    else:
        reward_terminal = 0.0
    return reward_terminal

def get_accum_reward_from_start(nn_map_len, nn_map_len_max, is_terminal):
    if FLAGS.reward_method == DEPTH:
        reward_accum = nn_map_len
    elif FLAGS.reward_method == EXP_DEPTH:
        reward_accum = (np.exp(np.log(FLAGS.reward_method_exp_coeff + 1) * nn_map_len / nn_map_len_max) - 1)
        if FLAGS.reward_method_normalize:
            reward_accum /= FLAGS.reward_method_exp_coeff
    elif FLAGS.reward_method == BINARY:
        reward_accum = 0.0
    else:
        assert False
    reward_accum += get_reward_terminal(nn_map_len, nn_map_len_max, is_terminal)
    return reward_accum

def get_reward_accumulated(nn_map_len_cur, nn_map_len_prev, nn_map_len_max, is_terminal):
    accum_reward_start2prev = get_accum_reward_from_start(nn_map_len_prev, nn_map_len_max, False)
    accum_reward_start2cur = get_accum_reward_from_start(nn_map_len_cur, nn_map_len_max, is_terminal)
    accum_reward_start2cur += get_reward_terminal(nn_map_len_cur, nn_map_len_max, is_terminal)
    return accum_reward_start2cur - accum_reward_start2prev

def get_reward_max(nn_map_len_max, is_terminal, nn_map_len_prev=None):
    accum_reward_start2prev = \
        get_accum_reward_from_start(0, nn_map_len_max, False) \
            if nn_map_len_prev is None else \
            get_accum_reward_from_start(nn_map_len_prev, nn_map_len_max, False)
    accum_reward_start2max = get_accum_reward_from_start(nn_map_len_max, nn_map_len_max, is_terminal)
    return accum_reward_start2max - accum_reward_start2prev


# except:
#     print('did not load seaborn')

def check_flags():
    # if FLAGS.node_feat_name:
    #     assert (FLAGS.node_feat_encoder == 'onehot')
    # else:
    #     assert ('constant_' in FLAGS.node_feat_encoder)
    # assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.layer_num >= 0)
    assert (FLAGS.batch_size >= 1)
    # assert (FLAGS.num_epochs >= 0)
    # assert (FLAGS.iters_val_start >= 1)
    # assert (FLAGS.iters_val_every >= 1)
    d = vars(FLAGS)
    ln = d['layer_num']
    ls = [False] * ln
    for k in d.keys():
        if 'layer_' in k and 'gc' not in k and 'branch' not in k and 'id' not in k:
            lt = k.split('_')[1]
            if lt != 'num':
                i = int(lt) - 1
                if not (0 <= i < len(ls)):
                    raise RuntimeError('Wrong spec {}'.format(k))
                ls[i] = True
    for i, x in enumerate(ls):
        if not x:
            raise RuntimeError('layer {} not specified'.format(i + 1))
    if 'cuda' in FLAGS.device:
        gpu_id = int(FLAGS.device.split(':')[1])
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError('Wrong GPU ID {}; {} available GPUs'.
                             format(FLAGS.device, gpu_count))
    # TODO: finish.


def get_flag(k, check=False):
    if hasattr(FLAGS, k):
        return getattr(FLAGS, k)
    else:
        if check:
            raise RuntimeError('Need flag {} which does not exist'.format(k))
        return None


def get_flags_with_prefix_as_list(prefix):
    rtn = []
    d = vars(FLAGS)
    i_check = 1  # one-based
    for k in sorted_nicely(d.keys()):
        v = d[k]
        sp = k.split(prefix)
        if len(sp) == 2 and sp[0] == '' and sp[1].startswith('_'):
            id = int(sp[1][1:])
            if i_check != id:
                raise ValueError('Wrong flag format {}={} '
                                 '(should start from _1'.format(k, v))
            rtn.append(v)
            i_check += 1
    return rtn


def load_replace_flags(pn):
    from saver import saver
    assert pn != 'None'
    loaded_flags = load(join(dirname(dirname(pn)), 'FLAGS.klepto'))['FLAGS']
    excluded_flags = {'device', 'MCTS', 'load_encoder', 'dataset', 'split_by', 'node_feats_for_mcs',
                      'node_feats_for_soft_mcs', 'tvt_options', 'num_iters',
                      'only_iters_for_debug', 'user', 'hostname', 'ts', 'train_subgroup_list',
                      'train_path_indices', 'train_subgroup', 'use_is_early_pruning',
                      'train_test_ratio', 'Q_BD', 'mcsplit_heuristic_on_iter_one',
                      'load_model', 'use_cached_gnn', 'long_running_val_mcsp',
                      'animation_size', 'recursion_threshold', 'promise_mode', 'regret_iters',
                      'explore_n_pairs', 'prune_n_bd_by_Q', 'save_every_recursion_count',
                      'eps_argmin', 'buffer_type', 'compute_loss_during_testing',
                      'debug_first_train_iters', 'restore_bidomains', 'no_pruning',
                      'mcsplit_heuristic_perc', 'populate_reply_buffer_every_iter',
                      'total_runtime', 'sample_all_edges', 'priority_correction',
                      'sample_all_edges_thresh', 'plot_final_tree', 'shuffle_input',
                      'time_analysis', 'no_search_tree', 'dataset_list', 'num_bds_max',
                      'num_nodes_max', 'val_every_iter', 'val_debug', 'plot_final_tree',
                      'drought_iters', 'val_method_list', 'beam_search', 'logging',
                      'num_bds_max', 'num_nodes_degree_max', 'randQ', 'DQN_mode',
                      'do_train', 'num_iters_threshold', 'timeout', 'subgroup',
                      'save_every', 'plot_tree', 'do_test', 'do_validation', 'model', 'mix',
                      'plot_solution', 'save_search', 'use_NN_for_u_score',
                      'show_precision','buffer_size','batch_size','print_buffer_stats',
                      'prune_trivial_rewards', 'print_rpreds_stats', 'num_epochs_per_learn',
                      'num_outer_epochs', 'regret_iters_train', 'show_precision', 'prune_trivial_rewards',
                      'mark_best_as_true', 'eps_decay_config', 'add_gnd_truth', 'regularization'
                      'train_dataset', 'train_subgroup', 'train_first_or_last',
                      'train_num_pairs', 'pretrain', 'imitation', 'num_iters_play',
                      'buffer_size', 'batch_size', 'print_buffer_stats',
                      'print_rpreds_stats', 'search_constraints', 'num_cs_refinements'}  # TODO: bug-prone here! list all related flags!
    diff_count = 0
    for k in vars(loaded_flags):
        if k in excluded_flags:
            continue
        if not hasattr(FLAGS, k):
            cur_v = None
        else:
            cur_v = getattr(FLAGS, k)
        loaded_v = getattr(loaded_flags, k)
        if cur_v != loaded_v:
            setattr(FLAGS, k, loaded_v)
            saver.log_info('\t{}={}\n\t\tto {}={}'.format(k, cur_v, k, loaded_v))
            diff_count += 1
            continue
    saver.log_info(f'Done loading FLAGS diff_count {diff_count}')
    # exit(-1)


def _get_layer_flags(layer_s):
    layer_split = layer_s.split(',')
    assert len(layer_split) >= 1
    rtn = OrderedDict()
    for s in layer_split:
        ss = s.split('=')
        assert len(ss) == 2
        rtn[ss[0]] = ss[1]
    return rtn


def _get_replaced_layer_flags(cur_l_flags, loaded_l_flags, excluded_flags):
    rtn = OrderedDict()
    for k in cur_l_flags:
        if k in excluded_flags:
            rtn[k] = cur_l_flags[k]
        elif k not in loaded_l_flags:
            rtn[k] = cur_l_flags[k]
        else:  # flags such as 'Q_mode'
            rtn[k] = loaded_l_flags[k]
    return rtn


def get_branch_names():
    bnames = get_flag('branch_names')
    if bnames:
        rtn = bnames.split(',')
        if len(rtn) == 0:
            raise ValueError('Wrong number of branches: {}'.format(bnames))
        return rtn
    else:
        assert bnames is None
        return None


def extract_config_code():
    with open(join(get_our_dir(), 'config.py')) as f:
        return f.read()


def convert_long_time_to_str(sec):
    def _give_s(num):
        return '' if num == 1 else 's'

    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} day{} {} hour{} {} min{} {:.3f} sec{}'.format(
        int(day), _give_s(int(day)), int(hour), _give_s(int(hour)),
        int(minutes), _give_s(int(minutes)), seconds, _give_s(seconds))


def get_our_dir():
    return cur_folder


def get_model_info_as_str():
    rtn = []
    d = vars(FLAGS)
    for k in d.keys():
        v = str(d[k])
        if k == 'dataset_list':
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
        else:
            vsplit = v.split('\n')
            assert len(vsplit) >= 1
            for i, vs in enumerate(vsplit):
                if i == 0:
                    ks = k
                else:
                    ks = ''
                if i != len(vsplit) - 1:
                    vs = vs + ','
                s = '{0:26} : {1}'.format(ks, vs)
                rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def get_model_info_as_command():
    rtn = []
    d = vars(FLAGS)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '--{}={}'.format(k, v)
        rtn.append(s)
    return 'python {} {}'.format(join(get_our_dir(), 'main.py'), '  '.join(rtn))


def debug_tensor(tensor, g1=None, g2=None):
    xxx = tensor.detach().cpu().numpy()
    # if g1 != None and g2 != None:
    #     import networkx as nx
    #     import matplotlib.pyplot as plt
    #     plt.subplot(121)
    #     nx.draw(g1, with_labels=True)  -
    #     plt.subplot(122)
    #     nx.draw(g2, with_labels=True)
    #     plt.show()
    return


TDMNN = None


# Let me know you question I turn on your voice now. ..............................................

def get_train_data_max_num_nodes(train_data):
    global TDMNN
    if TDMNN is None:
        TDMNN = train_data.dataset.stats['#Nodes']['Max']
    return TDMNN


def pad_extra_rows(g1x, g2x, padding_value=0):  # g1x and g2x are 2D tensors
    max_dim = max(g1x.shape[0], g2x.shape[0])

    x1_pad = torch.nn.functional.pad(g1x, (0, 0, 0, (max_dim - g1x.shape[0])),
                                     mode='constant',
                                     value=padding_value)
    x2_pad = torch.nn.functional.pad(g2x, (0, 0, 0, (max_dim - g2x.shape[0])),
                                     mode='constant',
                                     value=padding_value)

    return x1_pad, x2_pad


def plot_dist(data, label, save_dir, saver=None, analyze_dist=True, bins=None):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    fn = f'distribution_{label}.png'
    plt.figure()
    sns.set()
    ax = sns.distplot(data, bins=bins, axlabel=label)
    plt.xlabel(label)
    ax.figure.savefig(join(save_dir, fn))
    plt.close()


def _analyze_dist(saver, label, data):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Analyzing distribution of {label} (len={len(data)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    quantiles = mstats.mquantiles(data, prob=probs)
    func(f'{label} {len(data)}')
    s = '\t'.join([str(x) for x in probs])
    func(f'\tprob     \t {s}')
    s = '\t'.join(['{:.2f}'.format(x) for x in quantiles])
    func(f'\tquantiles\t {s}')
    func(f'\tnp.min(data)\t {np.min(data)}')
    func(f'\tnp.max(data)\t {np.max(data)}')
    func(f'\tnp.mean(data)\t {np.mean(data)}')
    func(f'\tnp.std(data)\t {np.std(data)}')


def plot_heatmap(data, label, save_dir, saver=None, analyze_dist=False):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Plotting heatmap of {label} (shape={len(data.shape)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    fn = f'heatmap_{label}.png'
    plt.figure()
    sns.heatmap(data, cmap='YlGnBu').figure.savefig(join(save_dir, fn))
    ax = sns.distplot(data, axlabel=label)
    plt.title(label)
    plt.close()


def plot_scatter_line(data_dict, label, save_dir):
    fn = f'scatter_{label}_iterations.png'
    ss = ['rs-', 'b^-', 'g^-', 'c^-', 'm^-', 'ko-', 'yo-']
    cs = [s[0] for s in ss]
    plt.figure()
    i = 0

    # min_size = min([len(x['incumbent_data']) for x in data_dict.values()])
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li, y_li = [], []

        # min_len = float('inf')
        # for x in data_dict_elt['incumbent_data']:
        #     if x[1] < min_len:
        #         min_len = x[1]

        for x in data_dict_elt['incumbent_data']:
            # if x[1] > FLAGS.recursion_threshold:
            #     break
            x_li.append(x[1])
            y_li.append(x[0])
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()

    plt.figure()
    fn = f'scatter_{label}_time.png'
    i = 0
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li = [x[2] for x in data_dict_elt['incumbent_data']]
        y_li = [x[0] for x in data_dict_elt['incumbent_data']]
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    # plt.close()


def print_g(g, label='', print_func=print):
    assert type(label) is str
    if label != '':
        label += ' '
    node_labels = set()
    for node, ndata in g.nodes(data=True):
        l = ndata.get('label')
        if l is not None:
            node_labels.add(l)
    print_func(f'{label}{g.number_of_nodes()} nodes {g.number_of_edges()} edges '
               f'(avg degree {g.number_of_edges() / g.number_of_nodes():.4f}) '
               f'{len(node_labels)} node labels')



from typing import Optional, Union, List
from collections import defaultdict
import torch
import torch_geometric.data


def pyg_from_networkx(G, group_node_attrs: Optional[Union[List[str], all]] = None,
                      group_edge_attrs: Optional[Union[List[str], all]] = None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """
    import networkx as nx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for key, value in data.items():
        try:
            data[key] = torch.tensor(value)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    if data.x is None:
        data.num_of_nodes = G.number_of_nodes()

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = [data[key] for key in group_node_attrs]
        xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        edge_attrs = [data[key] for key in group_edge_attrs]
        edge_attrs = [x.view(-1, 1) if x.dim() <= 1 else x for x in edge_attrs]
        data.edge_attr = torch.cat(edge_attrs, dim=-1)

    return data


from os.path import dirname, abspath, join


def get_save_path():
    return join(dirname(abspath(__file__)), 'save')


# def exec_cmd_timeout(cmd, timeout):
#     import subprocess
#     process = subprocess.call(cmd)
#     try:
#         process.wait(timeout=timeout)
#     except subprocess.TimeoutExpired:
#         process.terminate()


def exec_cmd_timeout(cmd, timeout):
    global exec_print
    import subprocess
    from threading import Timer
    from subprocess import Popen, PIPE

    if not timeout:
        process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        (std_output, std_error) = process.communicate()
        process.wait()
        # rc = process.returncode
        return std_output.decode("utf-8"), std_error.decode("utf-8"), False


    def kill_proc(proc, timeout_dict):
        timeout_dict["value"] = True
        proc.kill()

    def run(cmd, timeout_sec):
        proc = subprocess.Popen("exec " + cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        timeout_dict = {"value": False}
        timer = Timer(timeout_sec, kill_proc, [proc, timeout_dict])
        timer.start()
        stdout, stderr = proc.communicate()
        timer.cancel()
        return proc.returncode, stdout.decode("utf-8"), \
               stderr.decode("utf-8"), timeout_dict["value"]

    # if exec_print:
    #     print('Timed cmd {} sec(s) {}'.format(timeout, cmd))
    _, stdout, stderr, timeout_happened = run(cmd, timeout)
    # if exec_print:
    #     print('timeout_happened?', timeout_happened)
    return stdout, stderr, timeout_happened


