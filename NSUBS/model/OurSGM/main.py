from config import FLAGS, EXP_DEPTH
from saver import saver
from data_loader import get_data_loader_wrapper
from train import train
from test import test
from model_glsearch import GLS
from utils_our import load_replace_flags
from utils import OurTimer, save_pickle
from dvn_wrapper import create_dvn

import time
import torch
import traceback
import random, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
from os.path import join

class Logger():
    def __init__(self, search_num):
        self.search_num = search_num
        self.search_num2best_sln = {}

        self.is_exhausted_search_li = []
        self.precision_at_k = [0, 1e-12]
        self.precision_at_1 = [0, 1e-12]

        self.action_li_last = []
        self.search_num2diff_action_from_last = defaultdict(list)
        self.search_num2diff_action_from_true = defaultdict(list)

        self.search_num2Q_stats = {
            'pred_std': defaultdict(list),
            'true_std': defaultdict(list),
            'pred_true_correlation': defaultdict(list),
            'pred_true_MAE': defaultdict(list),
        }

        self.replay_buffer_sample_stats = {
            'nn_map_size': defaultdict(list),
            'action_space_size': defaultdict(list),
            'found_solution?': defaultdict(list),
            'max_reward': defaultdict(list),
            'std_Q_true': defaultdict(list),
            'epoch': defaultdict(list),
            'graph_pair': defaultdict(list)
        }

        self.search_num2buffer_size = defaultdict(list)
        self.search_num2rcs_perc_cs = defaultdict(list)
        self.common_target_nodes = defaultdict(lambda: 0)

        self.gid2iteration_time_best_nn_map = defaultdict(list)

    def save_search_curve(self):
        save_pickle(
            self.gid2iteration_time_best_nn_map,
            join(saver.get_log_dir(), 'search_curve.pkl'),
            print_msg=True
        )

    def reset_emb_logger(self):
        self.out_policy_logger = []
        self.out_value_logger = []

    def update_emb_logger_policy(self, node, pi):
        self.out_policy_logger.append({
            'policy_emb': node.bilin_emb,
            'is_best_action': [pi_elt == max(pi) for pi_elt in pi],
            'search_iter': len(node.nn_map)
        })

    def update_emb_logger_value(self, node, z, r_from_start):
        self.out_value_logger.append({
            'g_emb': node.g_emb,
            'r_from_start': r_from_start,
            'z': z
        })

    def plot_emb_logger(self, fn):
        tsne = TSNE()

        t0 = time.time()
        policy_emb_li, is_best_action_li, search_iter_li = [], [], []
        for policy_obj in self.out_policy_logger:
            policy_emb, is_best_action, search_iter = \
                policy_obj['policy_emb'], policy_obj['is_best_action'], policy_obj['search_iter']
            policy_emb_li.append(policy_emb)
            is_best_action_li.extend(is_best_action)
            search_iter_li.extend([search_iter]*len(is_best_action))

        if len(policy_emb_li) > 1:
            policy_emb_li = np.concatenate(policy_emb_li, axis=0)
            policy_emb_li = tsne.fit_transform(policy_emb_li)
            is_best_action_li = np.array(is_best_action_li)
            search_iter_li = np.array(search_iter_li)

            plt.title('policy embeddings')
            plt.scatter(
                policy_emb_li[is_best_action_li==True,0],
                policy_emb_li[is_best_action_li==True,1],
                c=search_iter_li[is_best_action_li==True],
                marker='x',
                cmap = 'plasma',
                alpha=0.5
            )
            plt.scatter(
                policy_emb_li[is_best_action_li==False,0],
                policy_emb_li[is_best_action_li==False,1],
                c=search_iter_li[is_best_action_li==False],
                marker='o',
                cmap = 'plasma',
                alpha=0.5
            )
            plt.colorbar()
            plt.savefig(join(saver.get_plot_dir(), f'tsne_policy_{fn}'))
            plt.close()
        saver.log_info(f'policy runtime: {time.time() - t0}s')

        t0 = time.time()
        g_emb_li, r_from_start_li, v_li = [], [], []
        for value_obj in self.out_value_logger:
            g_emb, r_from_start, v = \
                value_obj['g_emb'], value_obj['r_from_start'], value_obj['z']
            g_emb_li.append(g_emb)
            r_from_start_li.append(r_from_start)
            v_li.append(v)

        if len(g_emb_li) > 1:
            g_emb_li = np.stack(g_emb_li, axis=0)
            g_emb_li = tsne.fit_transform(g_emb_li)

            plt.title('value embeddings: V(s)')
            plt.scatter(
                g_emb_li[:,0],
                g_emb_li[:,1],
                c=np.array(v_li),
                cmap = 'plasma',
                norm = mpl.colors.LogNorm(),
                alpha=0.5
            )
            plt.colorbar()
            plt.savefig(join(saver.get_plot_dir(), f'tsne_value_{fn}'))
            plt.close()

            plt.title('value embeddings: cumulative reward')
            plt.scatter(
                g_emb_li[:,0],
                g_emb_li[:,1],
                c=np.array(r_from_start_li),
                cmap = 'plasma',
                norm = mpl.colors.LogNorm(),
                alpha=0.5
            )
            plt.colorbar()
            plt.savefig(join(saver.get_plot_dir(), f'tsne_cum_reward_{fn}'))
            plt.close()
        saver.log_info(f'value runtime: {time.time() - t0}s')

    def proc_x2y(self, x2y):
        x_li = []
        y_li = []
        for x in sorted(x2y.keys()):
            x_li.append(x)
            y_li.append(x2y[x])
        return x_li, y_li

    def proc_x2y_batch(self, x2y_batch):
        x_li = []
        y_li = []
        for x in sorted(x2y_batch.keys()):
            x_li.append(x)
            y_li.append(np.mean(np.array(x2y_batch[x])))
        return x_li, y_li

    def update_common_target_nodes(self, true_nn_map):
        if not FLAGS.plot_logs:
            return
        for v in true_nn_map.values():
            self.common_target_nodes[v] += 1

    def print_common_target_nodes(self):
        if not FLAGS.plot_logs:
            return
        if len(self.common_target_nodes) > 0:
            saver.log_info(f'num common_target_nodes: {len([u for u, count in self.common_target_nodes.items() if count > 1])}')
            saver.log_info(f'avg occurances target_nodes: {np.mean(np.array(list(self.common_target_nodes.values())))}')
            saver.log_info(f'max occurances target_nodes: {np.max(np.array(list(self.common_target_nodes.values())))}')

    def update_buffer_size(self, search_num, buffer_size):
        if not FLAGS.plot_logs:
            return
        self.search_num2buffer_size[search_num].append(buffer_size)

    def print_buffer_size(self):
        if not FLAGS.plot_logs:
            return
        search_num_li, buffer_size_li = self.proc_x2y_batch(self.search_num2buffer_size)
        plt.title('buffer size over time')
        plt.xlabel('search #')
        plt.plot(np.array(search_num_li), np.array(buffer_size_li))
        plt.savefig(f'{saver.get_plot_dir()}/logger_buffer_size.png')
        plt.close()

    def update_best_sln(self, search_num, best_found_solution):
        if not FLAGS.plot_logs:
            return
        self.search_num2best_sln[search_num] = [best_found_solution]

    def print_best_sln(self):
        if not FLAGS.plot_logs:
            return
        search_num_li, best_sln_li = self.proc_x2y_batch(self.search_num2best_sln)
        plt.title('best sln over time')
        plt.xlabel('search #')
        plt.plot(np.array(search_num_li), np.array(best_sln_li))
        plt.savefig(f'{saver.get_plot_dir()}/logger_best_sln_li.png')
        plt.close()

    def update_cs_bd_intersection(self, search_num, v_li, refined_v_li):
        if not FLAGS.plot_logs:
            return
        self.search_num2rcs_perc_cs[search_num].append(len(refined_v_li)/len(v_li))

    def print_cs_bd_intersection(self):
        if not FLAGS.plot_logs:
            return
        search_num_li, rcs_perc_cs_li = self.proc_x2y_batch(self.search_num2rcs_perc_cs)
        plt.title('|RCS|/|CS| over time')
        plt.xlabel('search #')
        plt.plot(np.array(search_num_li), np.array(rcs_perc_cs_li))
        plt.savefig(f'{saver.get_plot_dir()}/logger_cs_bd_intersection.png')
        plt.close()

    def update_replay_buffer_sample_stats(self, search_num, qsa, q_true):
        if not FLAGS.plot_logs:
            return
        self.replay_buffer_sample_stats['nn_map_size'][search_num].append(len(qsa.nn_map))
        self.replay_buffer_sample_stats['action_space_size'][search_num].append(len(qsa.v_li))

        assert FLAGS.reward_method == EXP_DEPTH
        self.replay_buffer_sample_stats['found_solution?'][search_num].append((q_true.max() > 99.9).detach().cpu().numpy())
        self.replay_buffer_sample_stats['max_reward'][search_num].append(q_true.max().detach().cpu().numpy())

        self.replay_buffer_sample_stats['std_Q_true'][search_num].append(torch.std(q_true).detach().cpu().numpy())
        # self.replay_buffer_sample_stats['epoch'][search_num].append(epoch)
        # self.replay_buffer_sample_stats['graph_pair'][search_num].append(mae)

    def print_replay_buffer_sample_stats(self):
        if not FLAGS.plot_logs:
            return
        plt.title('Replay buffer sample stats: |nn_map|')
        plt.xlabel('search #')
        search_num_li, nn_map_size_li = self.proc_x2y_batch(self.replay_buffer_sample_stats['nn_map_size'])
        plt.plot(np.array(search_num_li), np.array(nn_map_size_li), label='nn_map_size_li')
        plt.savefig(f'{saver.get_plot_dir()}/logger_rp_buffer_stats_nn_map.png')
        plt.close()

        plt.title('Replay buffer sample stats: |v_li|')
        plt.xlabel('search #')
        search_num_li, action_space_size_li = self.proc_x2y_batch(self.replay_buffer_sample_stats['action_space_size'])
        plt.plot(np.array(search_num_li), np.array(action_space_size_li), label='action_space_size_li')
        plt.savefig(f'{saver.get_plot_dir()}/logger_rp_buffer_stats_v_li.png')
        plt.close()

        plt.title('Replay buffer sample stats: % samples lead to found_solution')
        plt.xlabel('search #')
        search_num_li, found_solution_li = self.proc_x2y_batch(self.replay_buffer_sample_stats['found_solution?'])
        plt.plot(np.array(search_num_li), np.array(found_solution_li), label='found_solution_li')
        plt.savefig(f'{saver.get_plot_dir()}/logger_rp_buffer_stats_sln.png')
        plt.close()

        plt.title('Replay buffer sample stats: avg max_reward')
        plt.xlabel('search #')
        search_num_li, max_reward_li = self.proc_x2y_batch(self.replay_buffer_sample_stats['max_reward'])
        plt.plot(np.array(search_num_li), np.array(max_reward_li), label='max_reward_li')
        plt.savefig(f'{saver.get_plot_dir()}/logger_rp_buffer_stats_sln.png')
        plt.close()

        plt.title('Replay buffer sample stats: STD(Qtrue)')
        plt.xlabel('search #')
        search_num_li, std_Q_true_li = self.proc_x2y_batch(self.replay_buffer_sample_stats['std_Q_true'])
        plt.plot(np.array(search_num_li), np.array(std_Q_true_li))
        plt.savefig(f'{saver.get_plot_dir()}/logger_rp_buffer_stats_q.png')
        plt.close()

        # self.replay_buffer_sample_stats['epoch'][search_num].append(epoch)
        # self.replay_buffer_sample_stats['graph_pair'][search_num].append(mae)

    def update_q_pred_stats(self, search_num, q_pred, q_true):
        if not FLAGS.plot_logs:
            return
        pred_std = torch.std(q_pred).detach().cpu().numpy()
        true_std = torch.std(q_true).detach().cpu().numpy()
        idx_pred = torch.argmax(q_pred).detach().cpu().numpy()
        idx_true = torch.argmax(q_true).detach().cpu().numpy()
        correlation = idx_pred == idx_true
        mae = torch.mean(torch.abs(q_pred - q_true)).detach().cpu().numpy()

        self.search_num2Q_stats['pred_std'][search_num].append(pred_std)
        self.search_num2Q_stats['true_std'][search_num].append(true_std)
        self.search_num2Q_stats['pred_true_correlation'][search_num].append(correlation)
        self.search_num2Q_stats['pred_true_MAE'][search_num].append(mae)

    def print_q_pred_stats(self):
        if not FLAGS.plot_logs:
            return
        plt.title('Q vector stats: STD(Q)')
        plt.xlabel('search #')
        search_num_li, pred_std_li = self.proc_x2y_batch(self.search_num2Q_stats['pred_std'])
        search_num_li, true_std_li = self.proc_x2y_batch(self.search_num2Q_stats['true_std'])
        plt.plot(np.array(search_num_li), np.array(pred_std_li), label='pred_std_li')
        plt.plot(np.array(search_num_li), np.array(true_std_li), label='true_std_li')
        plt.legend()
        plt.savefig(f'{saver.get_plot_dir()}/logger_q_std.png')
        plt.close()

        plt.title('Q vector stats: MAE(Qpred, Qtrue)')
        plt.xlabel('search #')
        search_num_li, pred_true_MAE_li = self.proc_x2y_batch(self.search_num2Q_stats['pred_true_MAE'])
        plt.plot(np.array(search_num_li), np.array(pred_true_MAE_li))
        plt.savefig(f'{saver.get_plot_dir()}/logger_q_mae.png')
        plt.close()

        plt.title('Q vector stats: argmax(Qpred)==argmax(Qtrue)')
        plt.xlabel('search #')
        search_num_li, pred_true_correlation_li = self.proc_x2y_batch(self.search_num2Q_stats['pred_true_correlation'])
        plt.plot(np.array(search_num_li), np.array(pred_true_correlation_li))
        plt.savefig(f'{saver.get_plot_dir()}/logger_q_argmax.png')
        plt.close()

    def update_action_list(self, search_num, action_li, true_nn_map):
        if not FLAGS.plot_logs:
            return
        common_iterations = 0
        for i, (a_last, a) in enumerate(zip(self.action_li_last, action_li)):
            if a_last != a:
                common_iterations = i
                break
        self.search_num2diff_action_from_last[search_num].append(common_iterations)
        self.action_li_last = action_li

        if true_nn_map is not None and len(true_nn_map) > 0:
            common_iterations = 0
            for i, (u, v) in enumerate(action_li):
                if v != true_nn_map[u]:
                    common_iterations = i
                    break
            self.search_num2diff_action_from_true[search_num].append(common_iterations)

    def print_action_list(self):
        if not FLAGS.plot_logs:
            return
        plt.title('When search splits from "previous run"/"ground truth"')
        plt.xlabel('search #')
        search_num_li, diff_action_from_last_li = self.proc_x2y(self.search_num2diff_action_from_last)
        plt.plot(np.array(search_num_li), np.array(diff_action_from_last_li), label='diff_action_from_last')
        search_num_li, diff_action_from_true_li = self.proc_x2y(self.search_num2diff_action_from_true)
        plt.plot(np.array(search_num_li), np.array(diff_action_from_true_li), label='diff_action_from_true')
        plt.legend()
        plt.savefig(f'{saver.get_plot_dir()}/logger_which_action_differs.png')
        plt.close()

    def update_precision(self, precision_at_k, precision_at_1):
        self.precision_at_k[0] += precision_at_k[0]
        self.precision_at_k[1] += precision_at_k[1]
        self.precision_at_1[0] += precision_at_1[0]
        self.precision_at_1[1] += precision_at_1[1]

    def print_precision_and_reset(self):
        saver.log_info(f'precision@k: '
                       f'{self.precision_at_k[0] / self.precision_at_k[1]}, '
                       f'#samples={self.precision_at_k[1]}')
        saver.log_info(f'precision@1: '
                       f'{self.precision_at_1[0] / self.precision_at_1[1]}, '
                       f'#samples={self.precision_at_1[1]}')
        self.precision_at_k = [0, 1e-12]
        self.precision_at_1 = [0, 1e-12]

def get_d_in_raw():
    if FLAGS.do_train:
        dl = get_data_loader_wrapper('train')
    elif FLAGS.do_test:
        dl = get_data_loader_wrapper('test')
    else:
        assert False
    gp = next(dl)
    return gp.get_d_in_raw()

def main():
    if FLAGS.fix_randomness:
        saver.log_info('Critical! Fix random seed for torch and numpy')
        torch.manual_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        random.seed(FLAGS.random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(FLAGS.random_seed)


    model = _create_model(get_d_in_raw())
    # train_loader, test_loader = get_data_loader()
    assert not FLAGS.do_validation
    logger = Logger(0)
    if FLAGS.do_train:
        train(model, FLAGS.num_iters_threshold, FLAGS.timeout, logger)
    if FLAGS.do_test:
        with torch.no_grad():
            metrics = test(model, FLAGS.num_iters_threshold, FLAGS.timeout, logger, 'test')


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


if __name__ == '__main__':
    timer = OurTimer()

    try:
        if FLAGS.main_func == 'here':
            main()
        else:
            assert False
    except:
        traceback.print_exc()
        s = '\n'.join(traceback.format_exc(limit=-1).split('\n')[1:])
        saver.log_info(traceback.format_exc(), silent=True)
        saver.save_exception_msg(traceback.format_exc())
    finally:
        saver.log_info(f'Total time: {timer.time_and_clear()}')
        saver.close()

