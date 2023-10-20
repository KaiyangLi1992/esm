from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.search import SMTS, eval_print_search_result, logits2pi
from NSUBS.model.OurSGM.utils_our import get_reward_max
from NSUBS.model.OurSGM.config import FLAGS, BINARY, EXP_DEPTH
from NSUBS.model.OurSGM.saver import saver, get_our_dir
from NSUBS.src.utils import OurTimer, print_stats, format_seconds
from NSUBS.model.OurSGM.logger import EmbeddingLogger
from NSUBS.model.OurSGM.test import test

import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import join
from tqdm import tqdm
from operator import itemgetter

from collections import defaultdict
from enum import Enum

TRAIN_STAGE = Enum('', 'pretrain imitation normal')

def compute_eps(i):
    num_iters, start_eps, end_eps = \
        FLAGS.eps_decay_config['num_iters'], FLAGS.eps_decay_config['start_eps'], \
        FLAGS.eps_decay_config['end_eps']
    eps = max(end_eps, i / num_iters * (end_eps - start_eps) + start_eps)
    return eps


def daf_graph_assertion(gq, query_tree):
    print(sorted(list(gq.nx_graph.edges())))
    print(sorted(list([(key, sorted([child.nid for child in val.children])) for (key, val) in
                       query_tree.nid2node.items()])))
    for (nid1, nid2) in gq.nx_graph.edges():
        node1 = query_tree.nid2node[nid1]
        node2 = query_tree.nid2node[nid2]
        assert node2 in node1.children or node1 in node2.children
    print('query_tree validated')


import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return f'{size * 1e-9}gb'


class PretrainObj:
    def __init__(self, search_tree, solutions, cs_map, query_tree):
        self.search_tree = search_tree
        self.solutions = solutions
        self.cs_map = cs_map
        self.query_tree = query_tree

def increment_search_num(search_num, logger):
    search_num += 1
    logger.search_num = search_num
    print(f'search_num: {search_num}')
    if search_num % 100 == 0:
        print('saving logger figures...')
        logger.print_q_pred_stats()
        logger.print_action_list()
        logger.print_common_target_nodes()
        logger.print_best_sln()
        logger.print_buffer_size()
        logger.print_cs_bd_intersection()
        logger.print_replay_buffer_sample_stats()
        print('done')
    return search_num

def update_validation_set(model, logger, total_play_id, cur_best_metric, total_iter):
    if FLAGS.train_dataset == 'mix':
        # skip the validation step!
        cur_best_metric = 0.5
    else:
        t0 = time.time()
        val_loader = get_data_loader_wrapper('val')
        with torch.no_grad():
            metrics = test(model, FLAGS.num_iters_threshold_val, FLAGS.timeout_val, logger, total_play_id, test_loader=val_loader)
        metric_current = np.array(metrics['best_ratio_li']).mean()
        saver.writer.add_scalar('reward_validation', float(metric_current), total_iter)
        if len(metrics['best_ratio_li']) == 0 or metric_current < cur_best_metric:
            saver.log_info(f'REJECTING NEW MODEL: {metric_current} < {cur_best_metric}')
            ld = torch.load(join(saver.get_model_dir(), 'temp.pt'), map_location=FLAGS.device)
            model.load_state_dict(ld)
        else:
            saver.log_info(f'ACCEPTING NEW MODEL: {metric_current} >= {cur_best_metric}')
            cur_best_metric = metric_current
            torch.save(model.state_dict(), join(saver.get_model_dir(), f'best_{total_play_id}.pt'))
            torch.save(model.state_dict(), join(saver.get_model_dir(), 'best.pt'))
        saver.log_info(f'validation time: {time.time()-t0}s')
        model.train()
    return cur_best_metric

def train(model, num_iters_threshold, timeout, logger):
    model.train()
    assert FLAGS.buffer_size > FLAGS.batch_size, \
        saver.log_info(
            f'buffer must be larger than 1 batch: buffer_size={FLAGS.buffer_size}, batch_size={FLAGS.batch_size}')

    replay_buffer = []
    optimizer = optim.Adam(model.parameters())
    total_iter = 0
    search_num = 0
    total_play_id = 0
    cur_best_metric = update_validation_set(model, logger, total_play_id, -1, 0)
    for outer_epoch_id in range(FLAGS.num_outer_epoch):
        saver.log_info(f'Outer epoch {outer_epoch_id}')

        train_loader = get_data_loader_wrapper('train')
        for i, gp in enumerate(train_loader):
            embedding_logger = None

            timer = OurTimer()
            gp.to(FLAGS.device)
            saver.log_info(f'Loaded a graph pair {i}')
            # if i % FLAGS.save_every == 0:
            #     saver.save_trained_model(model, f'model_{outer_epoch_id}_{i}')

            gq, gt, CS, daf_path_weights, true_nn_map = gp.gq, gp.gt, gp.CS, gp.daf_path_weights, gp.true_nn_map
            # print('@@@ true_nn_map $$$', true_nn_map)
            eps = compute_eps(search_num)

            # CS, daf_path_weights, nn_map = preproc_graph_pair(gq.nx_graph, gt.nx_graph)
            # cached_tensors = model.cache_gq(gq.x, gq.edge_index)
            saver.log_info(f'Play the game {outer_epoch_id} -- '
                           f'{i} (eps={eps})')

            smts = SMTS(is_train=True, num_iters_threshold=num_iters_threshold, timeout=timeout)  # TODO: ask anonymous if it is okay to recreate it here
            if _get_stage(total_iter) == TRAIN_STAGE.imitation:
                override_use_heuristic = True
            else:
                override_use_heuristic = False

            timer.time_and_clear(f'play search starts', to_print=False)

            with torch.no_grad():
                scode = \
                    smts.search(
                        gq, gt, f'{outer_epoch_id}_{i}_train', model, CS, daf_path_weights, FLAGS.MCTS_train,
                        eps=eps, logger=logger,
                        override_use_heuristic=override_use_heuristic,
                        true_nn_map=true_nn_map
                    )  # TODO: give more running time for initial pretraining stage
            logger.update_common_target_nodes(gp.true_nn_map)
            search_num = increment_search_num(search_num, logger)
            eval_print_search_result(smts, gq)

            if scode == 2:  # error
                continue

            # print('@@@@@', len(smts.solutions))
            timer.time_and_clear(f'inner play search finishes', to_print=False)

            cur_name = getattr(gp, 'name', '')
            if cur_name != '':
                cur_name = ' ' + cur_name
            cur_name = f'{outer_epoch_id}_{i}{cur_name}'
            if FLAGS.plot_solution:
                saver.plot_solutions(smts, gq, gt, label=f'train_{cur_name}_sol',
                                     true_nn_map=gp.true_nn_map)
                timer.time_and_clear(f'play plot_solutions finishes', to_print=False)
            if FLAGS.plot_tree:
                saver.plot_search_tree(smts, label=f'train_{cur_name}_tree')
                timer.time_and_clear(f'play plot_tree finishes', to_print=False)
            if FLAGS.save_search:
                saver.save_search(smts, label=f'train_{cur_name}_smts')
                timer.time_and_clear(f'play save_search finishes', to_print=False)

            if FLAGS.print_buffer_stats:
                _print_qsa_li(smts.Qsa, replay_buffer)
            for Qsa in smts.Qsa:
                # if Qsa.u is None:
                #     print('&&&&&&')
                #     exit()
                Qsa.solutions = smts.solutions  # TODO: discussion: does every state need pretraining? maybe just iniital or GNN alone (without consensus)?
            replay_buffer.extend(smts.Qsa)
            # smts.clear()  # TODO potential bug: clear() misses something, e.g. timer object
            timer.time_and_clear(f'inner play end', to_print=False)

            replay_buffer = replay_buffer[-FLAGS.buffer_size:]
            random.shuffle(replay_buffer)

            # Learn!
            saver.log_info(f'Learn from buffer of {len(replay_buffer)}')
            pretrain_obj = PretrainObj(smts.search_tree, smts.solutions, CS[0], CS[1])
            saver.log_info(f'buffer_size={getsize(replay_buffer)}')
            saver.log_info(f'pretrain_obj={getsize(pretrain_obj)}')

            # save checkpoint
            torch.save(model.state_dict(), join(saver.get_model_dir(), 'temp.pt'))

            total_iter = learn(replay_buffer, model, optimizer, outer_epoch_id,
                               total_iter, pretrain_obj, logger, embedding_logger=embedding_logger)

            if total_play_id % FLAGS.val_every_games == 0:
                saver.log_info(f'Validation at self-play {total_play_id}')
                cur_best_metric = update_validation_set(model, logger, total_play_id, cur_best_metric, total_iter)

            timer.time_and_clear(f'Outer play learn finishes', to_print=False)
            # timer.print_durations_log(func=saver.log_info)
            total_play_id += 1

    logger.print_q_pred_stats()
    logger.print_action_list()
    logger.print_common_target_nodes()
    logger.print_cs_bd_intersection()
    logger.print_best_sln()
    logger.print_buffer_size()
    logger.print_replay_buffer_sample_stats()


def _print_qsa_li(Qsa_li, replay_buffer):
    r_li_lens, r_values, nn_map_sizes = [], [], []
    saver.log_info(f'len(Qsa_li)={len(Qsa_li)}')
    for qsa in Qsa_li:
        r_li_lens.append(len(qsa.v_r_accum_li_sampled))
        r_values.extend(qsa.v_r_accum_li_sampled)
        nn_map_sizes.append(len(qsa.nn_map))
    print_stats(r_li_lens, 'r_li_lens', print_func=saver.log_info)
    print_stats(r_values, 'r_values', print_func=saver.log_info)
    print_stats(nn_map_sizes, 'nn_map_sizes', print_func=saver.log_info)
    saver.log_info(f'replay_buffer len: {len(replay_buffer)}')


def _get_r_preds_X_q_X_t(qsa, model, embedding_logger):
    cs_map, query_tree = qsa.CS
    # print('@@@  embedding_logger embedding_logger', embedding_logger, 'model', model)
    # exit()
    model.reset_cache()
    assert model.dvn.encoder.Xq_Xt_cached_li is None
    out_policy, out_value, out_other = \
        model(
            qsa.gq, qsa.gt, qsa.u, qsa.v_li,
            qsa.nn_map, cs_map, qsa.candidate_map,
            FLAGS.cache_embeddings,
            graph_filter=qsa.graph_filter, filter_key=qsa.filter_key,
        )
    model.reset_cache()
    # assert v_li == qsa.v_li
    Xq, Xt = out_other['Xq'], out_other['Xt']
    return out_policy, out_value, Xq, Xt


# if FLAGS.glsearch:
#     loss_main = mse_loss(r_preds.view(-1), r_trues.view(-1))
#     # print(loss_main)
#     loss_reg = 0.0
# else:
#     # if True in torch.isinf(r_preds):
#     #     loss_main, loss_reg = 0.0, 0.0
#     # else:
def _loss_main(r_preds, r_trues, X_q, X_t, qsa):
    if FLAGS.loss_type == 'pos_neg_sampling' or FLAGS.loss_type == 'xentropy':
        loss_main = pos_neg_sampling_loss(r_preds, r_trues)
    elif FLAGS.loss_type == 'mse':
        loss_main = F.mse_loss(r_preds, r_trues)
    elif FLAGS.loss_type == 'mse_bounded':
        loss_main = mse_bounded(r_preds, r_trues, qsa)
        # print(f'r_preds:{r_preds}')
        # print(f'r_trues:{r_trues}')
    else:
        assert False
    return loss_main

def _get_loss(loss_main, loss_reg):
    return loss_main + FLAGS.regularization * loss_reg

def _backpropagate(loss_main_batch, loss_reg_batch, optimizer, retain_gradients):
    loss_batch = _get_loss(loss_main_batch, loss_reg_batch) # * batch_size / FLAGS.batch_size
    if loss_batch != 0:
        optimizer.zero_grad()
        loss_batch.backward(retain_graph=retain_gradients)
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()

def get_next_batch(replay_buffer, replay_buffer_iter, batch_size):
    replay_batch = replay_buffer[replay_buffer_iter: replay_buffer_iter + batch_size]
    # assert len(replay_batch) == batch_size, f'Wrong! will divide FLAGS,batch_size later but here ' \
    #                                         f'{len(replay_batch)} != {batch_size}'
    replay_buffer_iter += batch_size
    if replay_buffer_iter >= len(replay_buffer):
        replay_buffer_iter = 0
    return replay_batch, replay_buffer_iter


def _normal_stage_loss(replay_batch, epoch_iter, logger, embedding_logger, model):
    loss_main_batch, loss_reg_batch = \
        torch.tensor(0.0, device=FLAGS.device), torch.tensor(0.0, device=FLAGS.device)
    loss_dict = {}
    loss_dict['Policy Loss'] = 0
    loss_dict['Value Loss'] = 0
    loss_dict['MAE(V_pred, V_true)'] = 0
    loss_dict['argmax(p_pred) == argmax(p_true)'] = 0

    r_preds_batch = []

    # print('@@@ embedding_logger', embedding_logger, 'epoch_iter', epoch_iter)
    # exit()

    for i, qsa in enumerate(replay_batch):
        out_policy, out_value, Xq, Xt = _get_r_preds_X_q_X_t(qsa, model, embedding_logger)
        r_accum_true = torch.tensor(qsa.r_accum, dtype=torch.float32, device=FLAGS.device)

        r_preds_batch.extend(out_policy.tolist())
        if embedding_logger is not None and epoch_iter % 5 == 0 and i % 5 == 0:
            # assert FLAGS.reward_method == BINARY
            embedding_logger.buffer_entry_id = i
            embedding_logger.train_iter = epoch_iter
            embedding_logger.action_best_li = \
                [
                    f'{qsa.u}_{v}' for (i,v) in enumerate(qsa.v_li_sampled)
                    if qsa.v_r_accum_li_sampled[i] == qsa.v_r_accum_li_sampled.max()
                ]
        else:
            embedding_logger = None

        if FLAGS.dvn_config['decoder_policy']['type'] == 'GLSearch':
            assert not FLAGS.MCTS_train
            loss_main = F.mse_loss(out_policy, r_accum_true)
        else:
            if FLAGS.MCTS_train:
                pi_true = torch.tensor(logits2pi(qsa.v_pi_logits_li), dtype=torch.float32, device=FLAGS.device)
                # print(f'{out_policy},\t{pi_true}\n|\t{out_value},\t{r_accum_true}')
            else:
                idx, _ = max(enumerate(qsa.v_r_accum_li_sampled), key=itemgetter(1))
                idx_out_policy = qsa.v_li.index(qsa.v_li_sampled[idx])
                assert len(qsa.v_li) == out_policy.shape[0]
                # assert len(qsa.v_li_sampled) == len(qsa.v_r_accum_li_sampled) == len(qsa.v_pi_logits_li)
                pi_true = torch.zeros_like(out_policy, dtype=torch.float32, device=FLAGS.device)
                pi_true[idx_out_policy] = 1.0

            ce_loss = cross_entropy_smooth(out_policy, pi_true)
            mse_loss = 50 * mse_bounded(out_value, r_accum_true, qsa)
            loss_main = ce_loss + mse_loss
            loss_dict['Policy Loss'] += ce_loss.detach().cpu()
            loss_dict['Value Loss'] += mse_loss.detach().cpu()
            loss_dict['argmax(p_pred) == argmax(p_true)'] += np.argmax(
                out_policy.detach().cpu().numpy()) == np.argmax(pi_true.detach().cpu().numpy())

        loss_reg = _loss_regularized(Xq, Xt, qsa) if FLAGS.regularization > 1e-10 else 0.0

        loss_main_batch += loss_main
        loss_reg_batch += loss_reg
        # print(
        #     float(out_value.detach().cpu().numpy()),
        #     float(r_accum_true.detach().cpu().numpy()),
        #     list(out_policy.detach().cpu().numpy()),
        #     list(pi_true.detach().cpu().numpy()),
        #     float(mse_loss.detach().cpu().numpy()),
        #     float(ce_loss.detach().cpu().numpy()))
        loss_dict['MAE(V_pred, V_true)'] += torch.mean(torch.abs(out_value-r_accum_true)).detach().cpu()
        # timer.time_and_clear(f'get loss', to_print=True)
        # exit(-1)
    # timer.time_and_clear(f'loss/model iteration', to_print=True)
    assert len(replay_batch) > 0
    loss_main_batch /= len(replay_batch) + 1e-12
    loss_reg_batch /= len(replay_batch) + 1e-12
    loss_dict['Policy Loss'] /= len(replay_batch) + 1e-12
    loss_dict['Value Loss'] /= len(replay_batch) + 1e-12
    loss_dict['MAE(V_pred, V_true)'] /= len(replay_batch) + 1e-12
    loss_dict['argmax(p_pred) == argmax(p_true)'] /= len(replay_batch) + 1e-12
    return loss_main_batch, loss_reg_batch, r_preds_batch, loss_dict

def learn(replay_buffer, model, optimizer, outer_epoch_id,
          total_iter, pretrain_obj, logger, embedding_logger=None):
    timer = OurTimer()
    r_preds_list, used_QSAs = [], 0
    # total_iter = 0
    epoch_iter = 0
    replay_buffer_iter = 0
    # print('@@@ epoch_iter 1', epoch_iter)
    if logger is not None:
        logger.update_buffer_size(logger.search_num, len(replay_buffer))

    if len(replay_buffer) > 0:
        while epoch_iter < FLAGS.num_epochs_per_learn:
            stage = _get_stage(total_iter)
            if stage == TRAIN_STAGE.pretrain:
                # solutions from last search iteration
                # loss_main_batch, loss_reg_batch = \
                #     torch.tensor(0.0, device=FLAGS.device), torch.tensor(0.0, device=FLAGS.device)
                loss_dict = None
                loss_main_batch, loss_reg_batch = \
                    _pretrain_stage_loss(pretrain_obj.search_tree, optimizer, model,
                                         pretrain_obj.cs_map, pretrain_obj.query_tree,
                                         pretrain_obj.solutions, embedding_logger) # TODO: batch_size used here?
            elif stage in [TRAIN_STAGE.imitation, TRAIN_STAGE.normal]:
                replay_batch, replay_buffer_iter = \
                    get_next_batch(replay_buffer, replay_buffer_iter, FLAGS.batch_size)
                # print('@@@ epoch_iter 2', epoch_iter)
                loss_main_batch, loss_reg_batch, r_preds_batch, loss_dict = \
                    _normal_stage_loss(replay_batch, epoch_iter, logger, embedding_logger, model)
                r_preds_list.extend(r_preds_batch)  # logging
                _backpropagate(loss_main_batch, loss_reg_batch, optimizer, retain_gradients=True)
                epoch_iter += 1
            else:
                assert False

            # timer.time_and_clear(f'backprop', to_print=True)
            _log_loss(loss_dict, loss_main_batch, loss_reg_batch, outer_epoch_id, epoch_iter, total_iter,
                      stage)  # logging
            total_iter += 1
            # timer.time_and_clear(f'logging', to_print=True)
            # exit(-1)


            seconds = timer.get_duration()
            is_timeout = FLAGS.learning_timeout != -1 and seconds > FLAGS.learning_timeout
            # print('$$$$$$$', seconds, is_timeout, 'FLAGS.learning_timeout', FLAGS.learning_timeout)

            if is_timeout:
                saver.log_info(f'Learning: Time up! {format_seconds(seconds)}')
                return total_iter

        if FLAGS.print_rpreds_stats:
            print_stats(r_preds_list, 'r_preds_list', print_func=saver.log_info)
    return total_iter


def _log_loss(loss_dict, loss_main_batch, loss_reg_batch, outer_epoch_id, epoch_id, total_iter, stage):
    lm = float(loss_main_batch.detach().cpu())
    lr = float(loss_reg_batch.detach().cpu()) # DONT MULTIPLY BY REGULARIZATION HERE OR YOU DO IT TWICE!
    l = _get_loss(lm, lr)

    saver.writer.add_scalar('loss/loss_main', lm, total_iter)
    saver.writer.add_scalar('loss/loss_reg', lr, total_iter)

    saver.log_info(
        f'[Outer epoch {outer_epoch_id} Epoch {epoch_id} Iter {total_iter}] '
        f'{stage} loss: {l:.4f}\tmain/reg: {lm:.4f}/{lr:.4f}')
    saver.writer.add_scalar('loss/loss', l, total_iter)

    for loss_name, loss in loss_dict.items():
        saver.writer.add_scalar(f'loss/{loss_name}', float(loss), total_iter)


def _get_stage(total_iter):
    if FLAGS.pretrain:
        if FLAGS.imitation:
            if total_iter < FLAGS.pretrain_iters:
                return TRAIN_STAGE.pretrain
            elif total_iter < FLAGS.imitation_iters:
                return TRAIN_STAGE.imitation
            else:
                return TRAIN_STAGE.normal
        else:
            if total_iter < FLAGS.pretrain_iters:
                return TRAIN_STAGE.pretrain
            else:
                return TRAIN_STAGE.normal
    elif FLAGS.imitation:
        if total_iter < FLAGS.imitation_iters:
            return TRAIN_STAGE.imitation
        else:
            return TRAIN_STAGE.normal
    else:
        return TRAIN_STAGE.normal


def norm_r_trues(r_vec):
    return r_vec / r_vec.max()

def get_r_trues_regularization(u, v_li, solutions):
    solution_set = [v_li.index(solution[u]) for solution in solutions]
    true = torch.zeros(len(v_li), dtype=torch.float)
    true[solution_set] = 1
    true = torch.tensor(true, dtype=torch.float).to(FLAGS.device)
    return true

def cross_entropy_smooth(logits, pi):
    assert len(logits.shape) == 1
    pred = F.softmax(logits - logits.max()) + 1e-10
    return torch.sum(-pi * torch.log(pred))

def mse_bounded(pred, true, qsa, t=0.25):
    # print('@@@@@', max(pred))
    assert 0 <= t <= 1
    LB = true
    UB = \
        get_reward_max(qsa.gq.nx_graph.number_of_nodes(), True, nn_map_len_prev=len(qsa.nn_map)-1) * \
        torch.ones_like(LB, dtype=torch.float, device=FLAGS.device)
    assert False not in (LB <= UB)
    loss = torch.mean(torch.clamp(LB-pred, min=0.0)**2 + torch.clamp(pred-UB, min=0.0)**2)
    loss += t*F.mse_loss(pred, true)
    loss /= (1+t)
    return loss

def pos_neg_sampling_loss(pred, true):
    # assert FLAGS.reward_method == BINARY
    pos_samples = pred[true == true.max()]
    neg_samples = pred[true != true.max()]
    # print(f'(+) pair - avg:{pos_pairs.mean()}\tmax:{pos_pairs.max()}\tmin:{pos_pairs.min()}')
    # print(f'(-) pair - avg:{neg_pairs.mean()}\tmax:{neg_pairs.max()}\tmin:{neg_pairs.min()}')
    # print('=====================================')
    assert pos_samples.shape[0] > 0 and neg_samples.shape[0] > 0
    N, = pos_samples.shape
    ce_samples = torch.cat([pos_samples.view(N, 1), neg_samples.repeat(N, 1)], dim=1)
    loss = F.cross_entropy(ce_samples, torch.zeros(N, dtype=torch.long, device=FLAGS.device))
    return loss

# def sigmoid_inv(x, clip=None):
#     if clip is None:
#         out = -torch.log((1 / (x + 1e-8)) - 1)
#     else:
#         clip = torch.abs(clip.detach())
#         out = torch.clip(-torch.log((1 / (x + 1e-8)) - 1), -clip, clip)
#     return out

# pred = pred - torch.mean(pred).detach()
# pred = pred / (torch.abs(pred).max().detach() + 1e-4) * 10
# pred_norm = pred + sigmoid_inv(torch.mean(true))
def _loss_regularized(X_q, X_t, qsa):
    return 0.0
    cs_map, _ = qsa.CS
    solutions = qsa.solutions

    if len(solutions) == 0:
        return 0 # TODO: in some extreme case, no solutions
    matching_matrix = torch.matmul(X_q, torch.transpose(X_t, 1, 0))

    loss_total, loss_norm = 0.0, 1e-8
    for u, v_li in cs_map.items():
        if len(v_li) > 1:
            pred = matching_matrix[[u] * len(v_li), v_li]
            true = get_r_trues_regularization(u, v_li, solutions)
            if torch.abs(true.max()-true.min()) > 1e-5:
                loss_total += pos_neg_sampling_loss(pred, true)
                loss_norm += 1
    loss_reg = loss_total / loss_norm

    return loss_reg

def _pretrain_stage_loss(search_tree, optimizer, model, cs_map, query_tree, solutions,
                         embedding_logger):
    num_samples_batch = 0
    loss_main_batch, loss_reg_batch = \
        torch.tensor(0.0, device=FLAGS.device), torch.tensor(0.0, device=FLAGS.device)
    # print('@@@ len(solutions)', len(solutions), 'solutions', solutions)
    # exit(-1)
    for z, solution in enumerate(solutions):
        num_samples = 0
        loss_main = torch.tensor(0.0, device=FLAGS.device)
        state = search_tree.root
        # print('@@@ len(state.v_li)', len(state.v_li))
        # exit(-1)
        while len(state.v_li) > 0:
            v_pos = solution[state.u]
            if len(state.v_li) > 1:
                num_samples += 1
                loss_main += \
                    _compute_pretrain_loss(v_pos, state, model, cs_map, query_tree, embedding_logger)
                # print(f'solution={z} sample={num_samples}')
                # os.system('nvidia-smi')
                # print(f'v_pos={getsize(v_pos)}, state={getsize(state)}, model={getsize(model)}, cs_map={getsize(cs_map)}, query_tree={getsize(query_tree)}')
            state = get_next(search_tree, state, v_pos)
        _backpropagate(loss_main / num_samples, loss_reg_batch, optimizer, retain_gradients=True)
        loss_main_batch += loss_main.detach()
        # print('loss_main_batch', loss_main_batch)
        num_samples_batch += num_samples
        # print('num_samples_batch', num_samples_batch)
    return loss_main_batch / (1e-8 + num_samples_batch), loss_reg_batch


def _compute_pretrain_loss(v_pos, state, model, cs_map, query_tree, embedding_logger):
    assert all([Xt_cached is None for Xt_cached in model.dvn.encoder.Xt_cached])
    _, _, out_other = \
        model(
            state.gq, state.gt, state.u, state.v_li,
            state.nn_map, cs_map, query_tree,
            graph_filter=None, filter_key=None
        )
    X_q, X_t = out_other['Xq'], out_other['Xt']
    if embedding_logger is not None:
        embedding_logger.log_embeddings_tensorboard(X_q, X_t)
    # neg_samples = list(random.sample(set(state.v_li) - {v_pos}, min(10, len(state.v_li)-1)))
    # ce_samples = torch.matmul(X_q[state.u].view(1,-1), X_t[[v_pos] + neg_samples].transpose(0,1))
    # loss = F.cross_entropy(ce_samples, torch.tensor([0], dtype=torch.long, device=FLAGS.device))
    ce_samples = torch.matmul(X_q[state.u].view(1, -1), X_t[state.v_li].transpose(0, 1))
    loss = F.cross_entropy(ce_samples, torch.tensor([state.v_li.index(v_pos)], dtype=torch.long,
                                                    device=FLAGS.device))
    # print(f'ce_samples {ce_samples}')
    # print(f'true {state.v_li.index(v_pos)}')
    # print(f'loss {loss}')
    # exit(-1)
    return loss


def get_next(search_tree, state, v_pos):
    next_state = None
    for nid_child in search_tree.nxgraph.neighbors(state.nid):
        if search_tree.nxgraph.edges[(state.nid, nid_child)]['action'] == (state.u, v_pos):
            next_state = search_tree.nid2node[nid_child]
            break
    assert next_state is not None
    return next_state
