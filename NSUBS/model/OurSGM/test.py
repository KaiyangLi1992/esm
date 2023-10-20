from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver
from NSUBS.src.utils import print_stats, format_seconds

from NSUBS.model.OurSGM.utils_our import get_reward_accumulated

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# from common import utils
# from subgraph_matching.config import parse_encoder
# from subgraph_matching.train import build_model
from os.path import join
import argparse
import random

import torch.nn.functional as F
import numpy as np
import torch
from NSUBS.model.OurSGM.search import SMTS, eval_print_search_result
from NSUBS.model.OurSGM.daf import preproc_graph_pair

PRINT_TSNE = False

def test(model, num_iters_threshold, timeout, logger, label='', test_loader=None):
    if test_loader is None:
        test_loader = get_data_loader_wrapper('test')
    if model is not None:
        model.eval()
    num_solutions_li = []
    num_unique_node_set_solutions_li = []
    best_size_li = []
    best_ratio_li = []
    best_iter_if_found_li = []
    total_time_li = []
    solved_or_not_li = []
    total_iters_li = []

    nsm_model = None
    if FLAGS.model == 'nsm':
        parser = argparse.ArgumentParser(description='Alignment arguments')
        utils.parse_optimizer(parser)
        parse_encoder(parser)
        args = parser.parse_args()
        args.test = True
        nsm_model = build_model(args)
        saver.log_info(f'Loaded nsm model {nsm_model}')
    # else:
    #     exit()

    embedding_dict = {}
    for i, gp in enumerate(test_loader):
        gp.to(FLAGS.device)

        if FLAGS.model == 'nsm':
            saver.log_info(f'Running nsm model...')
            # align_matrix = gen_alignment_matrix(nsm_model, gp.gq.nx_graph, gp.gt.nx_graph, method_type="order")
            # align_matrix = align_matrix.detach().cpu().numpy()
            # print('align_matrix.shape', align_matrix.shape)
            # print('align_matrix', align_matrix)
            # exit()
            align_matrix = nsm_model
        else:
            align_matrix = None


        smts = SMTS(is_train=False, num_iters_threshold=num_iters_threshold, timeout=timeout)
        # nn_map = {}
        gq, gt, CS, daf_path_weights = gp.gq, gp.gt, gp.CS, gp.daf_path_weights
        # CS, daf_path_weights, nn_map = preproc_graph_pair(gq.nx_graph, gt.nx_graph)
        # if FLAGS.matching_order == 'nn':
        #     cached_tensors = model.cache_gq(gq.x, gq.edge_index)
        # else:

        with torch.no_grad():
            smts.search(gq, gt, f'{i}_test', model, CS, daf_path_weights, FLAGS.MCTS_test, logger=logger, align_matrix=align_matrix)

        logger.save_search_curve()

        num_solutions, num_solutions_unique, best_size, best_ratio = \
            eval_print_search_result(smts, gq)

        num_solutions_li.append(num_solutions)
        num_unique_node_set_solutions_li.append(num_solutions_unique)
        best_size_li.append(best_size)
        best_ratio_li.append(best_ratio)

        if PRINT_TSNE and FLAGS.matching_order == 'nn':
            embedding_dict[i] = \
                {
                    'g_emb':
                        [
                            (
                                node.g_emb,
                                get_reward_accumulated(
                                    node.max_depth, len(node.nn_map)-1,
                                    smts.search_tree.root.gq.nx_graph.number_of_nodes(), True
                                ), node.g_emb
                            )
                            for node in random.sample(smts.search_tree.nid2node.values(), k=min(len(smts.search_tree.nid2node),100)) if node.g_emb is not None
                        ],
                    'intra_emb':
                        (torch.sum(model.dvn.encoder.Xq_Xt_cached_li[-1][0], dim=0).detach().cpu().numpy(), num_solutions)
                }

        if num_solutions >= 1:
            best_iter_if_found_li.append(smts.best_nn_map_iter)

        saver.log_info(f'time: {format_seconds(smts.total_time)}')

        total_time_li.append(smts.total_time)

        solved_or_not_li.append(1 if num_solutions != 0 else 0)

        total_iters_li.append(smts.num_iters)

        if FLAGS.plot_solution:
            saver.plot_solutions(smts, gq, gt, label=f'val_{label}_{i}_sol',
                                 true_nn_map=gp.true_nn_map, CS=CS)

        if FLAGS.plot_tree:
            saver.plot_search_tree(smts, label=f'val_{label}_{i}_tree')

        if FLAGS.save_search:
            saver.save_search(smts, label=f'val_{label}_{i}_smts')

        print_stats(solved_or_not_li, 'solved_or_not_li', print_func=saver.log_info)
        print_stats(num_solutions_li, 'num_solutions_li', print_func=saver.log_info)
        print_stats(num_unique_node_set_solutions_li, 'num_unique_node_set_solutions_li',
                    print_func=saver.log_info)
        print_stats(best_size_li, 'best_size_li', print_func=saver.log_info)
        print_stats(best_ratio_li, 'best_ratio_li', print_func=saver.log_info)
        print_stats(best_iter_if_found_li, 'best_iter_if_sol_is_found_li', print_func=saver.log_info)
        print_stats(total_time_li, 'total_time_li(in seconds)', print_func=saver.log_info)
        print_stats(total_iters_li, 'total_iters_li', print_func=saver.log_info)

    metrics = {
        'solved_or_not_li': solved_or_not_li,
        'num_solutions_li': num_solutions_li,
        'best_size_li': best_size_li,
        'best_ratio_li': best_ratio_li,
        'best_iter_if_found_li': best_iter_if_found_li,
        'total_time_li': total_time_li,
        'total_iters_li': total_iters_li
    }

    if PRINT_TSNE and FLAGS.matching_order == 'nn':
        print_tsne(embedding_dict)

    return metrics

def print_tsne(embedding_dict):
    saver.save_as_pickle(embedding_dict, 'emebdding_dict')
    X_value = []
    y_value = []
    X_intra = []
    y_intra = []
    for metadata in embedding_dict.values():
        if len(metadata['g_emb']) > 0:
            X_value.append(np.stack([x[0] for x in metadata['g_emb']],axis=0))
            y_value.append(np.stack([x[1] for x in metadata['g_emb']],axis=0))
        X_intra.append(metadata['intra_emb'][0])
        y_intra.append(metadata['intra_emb'][1])
    X_value = np.concatenate(X_value, axis=0)
    y_value = np.concatenate(y_value, axis=0)
    X_intra = np.stack(X_intra, axis=0)
    y_intra = np.array(y_intra)
    print(X_value.shape)
    print(y_value.shape)
    print(X_intra.shape)
    print(y_intra.shape)
    plot_points(X_value, y_value, title='value')
    plot_points(X_intra, y_intra, title='intra')

def plot_points(X, y, title='plot'):
    Z = TSNE(n_components=2).fit_transform(X)
    plt.figure(figsize=(8, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=20, c=y)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(join(saver.get_plot_dir(), f'{title}.png'))
    plt.close()
