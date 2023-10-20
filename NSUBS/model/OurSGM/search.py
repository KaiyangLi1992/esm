from NSUBS.model.OurSGM.config import FLAGS, ADD_GND_TRUTH_AT_START, ADD_GND_TRUTH_AT_END, INDUCED, ISOMORPHIC, HOMOMORPHIC
from NSUBS.model.OurSGM.saver import saver

from NSUBS.src.utils import OurTimer, format_seconds, print_stats
import itertools
import collections
import numpy as np
import math

from os.path import join
from copy import deepcopy
from NSUBS.model.OurSGM.data_structures_common import QsaObj, GlobalSearchParams
from NSUBS.model.OurSGM.search_tree import SearchTree, SearchNode, BFS_ORDERING
from NSUBS.model.OurSGM.utils_our import get_reward_accumulated, get_accum_reward_from_start
try:
    from NSUBS.model.OurSGM.nsm.alignment import gen_alignment_matrix
except:
    print('not using nsm!')
from NSUBS.model.OurSGM.dvn_wrapper import GraphFilter

import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_scatter import scatter_add


class StreamingPlotter():
    def __init__(self, n, stream_name_li):
        self.n = n
        self.stream_dict = {stream_name:{'t':[], 'y':[], 'frozen': False} for stream_name in stream_name_li}
        pass

    def add_point(self, stream_name, t, y):
        stream = self.stream_dict[stream_name]
        if not stream['frozen']:
            if len(stream['y']) >= 2 and (stream['y'][-1] == y and stream['y'][-2] == y):
                self.stream_dict[stream_name]['t'].pop()
                self.stream_dict[stream_name]['y'].pop()

            self.stream_dict[stream_name]['t'].append(t)
            self.stream_dict[stream_name]['y'].append(y)

            if len(self.stream_dict[stream_name]['t']) >= self.n:
                self.stream_dict[stream_name]['frozen'] = True

    def plot_streams(self, pn, title, x_label, y_label):
        print('plotting search process...')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        for stream_name, stream in self.stream_dict.items():
            plt.plot(stream['t'], stream['y'], label=stream_name, alpha=0.65)
        plt.legend()
        plt.savefig(pn)
        plt.close()
        print('plotting done...')

class SearchStreamPlotLogger():
    def __init__(self):
        self.NN_MAP_BEST = 'nn_map_best'
        self.NN_MAP = 'nn_map'
        self.streaming_plotter_time = None
        self.streaming_plotter_iteration = None

    def reset(self):
        self.streaming_plotter_time = StreamingPlotter(10_000, (self.NN_MAP_BEST, self.NN_MAP))
        self.streaming_plotter_iteration = StreamingPlotter(10_000, (self.NN_MAP_BEST, self.NN_MAP))

    def add_point(self, nn_map_best, nn_map, time, iteration):
        self.streaming_plotter_time.add_point(self.NN_MAP_BEST, time, nn_map_best)
        self.streaming_plotter_time.add_point(self.NN_MAP, time, nn_map)
        self.streaming_plotter_iteration.add_point(self.NN_MAP_BEST, iteration, nn_map_best)
        self.streaming_plotter_iteration.add_point(self.NN_MAP, iteration, nn_map)

    def plot_streams(self, pn):
        self.streaming_plotter_time.plot_streams(f'{pn}_time.png', 'search process [time]', 'time', '|subgraph|')
        self.streaming_plotter_iteration.plot_streams(f'{pn}_iter.png', 'search process [iterations]', 'iterations', '|subgraph|')

def get_u_next_li(gq, query_tree, nn_map):
    u_next_li = \
        list(
            {u for u in gq.nodes() if
             set(query_tree.nid2node[u].get_parent_nids()).issubset(set(nn_map.keys()))} \
            - set(nn_map.keys())
        )
    # print(f'8: {query_tree.nid2node[8].get_parent_nids()}')
    # print(f'3: {query_tree.nid2node[3].get_parent_nids()}')
    # print(f'keys: {set(nn_map.keys())}')
    return u_next_li

def get_v_candidates_wrapper(u, nn_map, gq, gt, cs_map, query_tree):
    is_root = is_u_a_root_node(query_tree, u)
    if is_root:  # base case: root node
        v_candidates = cs_map[u]
    else:
        v_candidates = get_v_candidates(u, nn_map, gq, gt, cs_map, query_tree)
    return v_candidates, is_root

def is_u_a_root_node(query_tree, u):
    return len(query_tree.nid2node[u].get_parent_nids()) == 0

def get_v_candidates(u, nn_map, gq, gt, cs_map, query_tree):
    assert FLAGS.search_constraints in {INDUCED, ISOMORPHIC, HOMOMORPHIC}

    # if FLAGS.time_analysis:
    #     timer.time_and_clear(f'get_v_candidates {random.random()} {u}')


    u_parents = query_tree.nid2node[u].get_parent_nids()

    # if FLAGS.time_analysis:
    #     timer.time_and_clear(f'get_parent_nids {random.random()} {u}')

    assert len(u_parents) > 0
    v_parents = [nn_map[u_parent] for u_parent in u_parents]

    # if FLAGS.time_analysis:
    #     timer.time_and_clear(f'v_parents {random.random()} {u}')

    # find all nodes in v that have v_parents
    # f(*[a,b,c,d]) == f(a,b,c,d)
    v_candidates = list(
        set(cs_map[u]).intersection(
            *[set(gt.neighbors(v_parent)) for v_parent in v_parents]))

    # if FLAGS.time_analysis:
    #     timer.time_and_clear(f'set(cs_map[u]).intersection {random.random()} {u} {len(v_parents)} {len(v_candidates)} {len(cs_map[u])}')
    '''
    redundant: nodes must connect only to v_parents in nn_map and 
    not other nodes in nn_map thats not v_parents
    '''
    if FLAGS.search_constraints == INDUCED:
        v_candidates = [v for v in v_candidates if
                        set.isdisjoint(set(gt.neighbors(v)),
                                       set(nn_map.values()) - set(v_parents))]

    # if FLAGS.time_analysis:
    #     timer.time_and_clear(f'search_constraints == INDUCED {random.random()} {u}')
    if FLAGS.search_constraints != HOMOMORPHIC:
        v_candidates = [v for v in v_candidates if v not in nn_map.values()]

    # if FLAGS.time_analysis:
    #     timer.time_and_clear(f'search_constraints != HOMOMORPHIC {random.random()} {u}')
    # for v in v_candidates:
    #     print([set(gt.neighbors(v_parent)) for v_parent in v_parents])
    #     print(set(cs_map[u]))
    #     print(set.isdisjoint(set(gt.neighbors(v)), set(nn_map.values()) - set(v_parents)))
    #     assert show_solution_temp(gq, gt, nn_map, u_v=(u, v))
    return v_candidates

def show_solution_temp(gq, gt, nn_map, u_v=None):
    if u_v is not None:
        u, v = u_v
        nn_map = deepcopy(nn_map)
        nn_map[u] = v
    print(f'nn_map size: {len(nn_map)}')
    gq = nx.subgraph(gq, nn_map.keys())
    gt = nx.subgraph(gt, nn_map.values())
    gq = nx.relabel_nodes(gq, nn_map)
    gq_edges = set([tuple(sorted([u, v])) for (u, v) in gq.edges() if u != v])
    gt_edges = set([tuple(sorted([u, v])) for (u, v) in gt.edges() if u != v])
    print(gq_edges)
    print(gt_edges)
    return gq_edges == gt_edges

def logits2pi(logits):
    temp = FLAGS.MCTS_temp
    logits = np.array(logits)
    if temp is None:
        pi = np.zeros_like(logits)
        pi[np.random.choice(np.flatnonzero(logits == logits.max()))] = 1
    else:
        logits = [x ** (1. / FLAGS.MCTS_temp) for x in logits]
        counts_sum = float(sum(logits)) + 1e-12
        pi = np.array([x / counts_sum for x in logits])
        # pi = np.exp(logits/temp)/np.sum(np.exp(logits/temp) + 1e-12)
    return pi

class SMTS():
    def __init__(self, is_train, num_iters_threshold, timeout):
        self.Qsa = []  # stores Q values for s,a (as defined in the paper)
        self.solutions = []
        self.num_iters = 0
        self.num_iters_threshold = num_iters_threshold
        self.time_seconds_threshold = timeout
        self.timer = OurTimer()  # start timing here!
        self.debugger_timer = None
        if FLAGS.time_analysis:
            self.debugger_timer = OurTimer()
        self.total_time = 0

        self.best_reward_accum = -float('inf')
        self.best_nn_map = {}
        self.best_nn_map_iter = -1

        self.is_train = is_train
        self.search_tree = None  # SearchTree(root, self.is_train)
        # self.search_stream_plot_logger = SearchStreamPlotLogger()

        self.nn_call_cnt = 0
        self.eps_call_cnt = 0
        self.eps_cond1_call_cnt = 0
        self.eps_cond2_call_cnt = 0
        self.eps_cond3_call_cnt = 0
        self.eps_cond4_call_cnt = 0

        self.MCTS_iter = 0
        self.MCTS_Qs = []
        self.MCTS_Us = []
        self.MCTS_QU = []

    def search(self, gq, gt, gid, model, CS, daf_path_weights, MCTS, eps=0.0, logger=None,
               override_use_heuristic=False, true_nn_map=None, align_matrix=None):#, skip_if_action_space_less_than=None):
        if model is not None:
            model.reset_cache()
        rtn = \
            self._search(
                gq, gt, gid, model, CS, daf_path_weights, MCTS, eps=eps, logger=logger,
                override_use_heuristic=override_use_heuristic,
                true_nn_map=true_nn_map, align_matrix=align_matrix
            )
        # if model is not None:
        #     model.reset_cache()
        # print('Here!!!!!')
        # exit()
        self.total_time = self.timer.get_duration()

        if len(self.solutions) == 0:
            print(f'graph pair {gid} could not find the solution!')
        return rtn
        # except TimeoutError:
        #     self.total_time = self.timer.get_duration()
        #     return 1
        # except RuntimeError as e:
        #     saver.log_info(str(e))
        #     return 2

    def _search(self, gq, gt, gid, model, CS, daf_path_weights, MCTS, eps=0.0, logger=None,
                override_use_heuristic=False, true_nn_map=None, align_matrix=None):
        if logger is not None and self.is_train:
            logger.reset_emb_logger()

        graph_filter = GraphFilter()
        global_search_params = \
            GlobalSearchParams(
                gq, gt, CS, daf_path_weights, model, graph_filter,
                override_use_heuristic, eps, logger, gid, MCTS, align_matrix
            )

        if gt.edge_index.shape[0] == 0:
            print(f'CS:{CS}')
            print(f'gt:{gt.nx_graph.number_of_nodes()}, {gt.edge_index.shape},\n{gt.edge_index}')
            print(f'gq:{gq.nx_graph.number_of_nodes()}, {gq.edge_index.shape},\n{gq.edge_index}')
            raise RuntimeError(f'encoder() prunes gt to be empty :(')

        # self.search_stream_plot_logger.reset()
        root = self._create_root(global_search_params, {})
        if root is not None:
            self.search_tree = SearchTree(root, self, self.is_train, cs_map=global_search_params.CS[0])
            # print('@@@ true_nn_map', true_nn_map)
            pop_method = self._add_pseudo_gnd_truth_at_start(true_nn_map, global_search_params)

            action_li = []
            while self._check_exit_conditions():

                if FLAGS.time_analysis:
                    self.debugger_timer.reset()

                cur_node = self.search_tree.get_node(pop_method)

                if FLAGS.time_analysis:
                    self.debugger_timer.time_and_clear('search_tree.get_node')

                r = self._run_single_iteration(cur_node, global_search_params)

                if FLAGS.time_analysis:
                    self.debugger_timer.time_and_clear('_run_single_iteration')

                if r is None:
                    break

                u, v, next_node = r
                self.search_tree.nxgraph.nodes[next_node.nid]['search_stage'] = 'outer' # for MCSP logging!
                assert u is not None and v is not None

                # From here, we treat cur_node as pseduo-root and do MCTS.
                pop_method = self.search_tree.update_search_stack(cur_node, next_node, pop_method)
                # logging
                action_li.append((u,v))
                # self._log_nn_map_time_iter_tuple(next_node)
                self.num_iters += 1
                # print('self._check_exit_conditions()?', self._check_exit_conditions())

                if FLAGS.time_analysis:
                    self.debugger_timer.print_durations_log()
            if self.is_train:
                # print('@@@ true_nn_map 2', true_nn_map)
                # self._add_pseudo_gnd_truth_at_end(true_nn_map, global_search_params)
                if FLAGS.glsearch:
                    self.Qsa.extend(
                        self.search_tree.get_QsaObj_li_glsearch(
                            global_search_params, graph_filter)
                    )
                else:
                    self.Qsa.extend(
                        self.search_tree.get_QsaObj_li(
                            global_search_params, graph_filter if self.is_train else None)
                    )
            if logger is not None and self.is_train:
                logger.update_best_sln(logger.search_num, len(self.best_nn_map))
                logger.update_action_list(logger.search_num, action_li, true_nn_map)
                logger.is_exhausted_search_li.append(self.search_tree.check_empty())
        else:
            if logger is not None and self.is_train:
                # if root is empty -> trivial pair -> search_tree done though
                logger.is_exhausted_search_li.append(True)

        # if logger is not None and self.is_train:
        #     logger.plot_emb_logger(gid)
        # if not self.is_train:
        #     self.search_stream_plot_logger.plot_streams(join(saver.get_plot_dir(), f'stream_{gid}.png'))

    def _add_pseudo_gnd_truth_at_start(self, true_nn_map, global_search_params):
        # print('true_nn_map', true_nn_map, 'len(true_nn_map) != 0', len(true_nn_map) != 0)
        if self.is_train and FLAGS.add_gnd_truth == ADD_GND_TRUTH_AT_START and self.is_train and (true_nn_map is not None and len(true_nn_map) != 0):
            # print('@@@@@@@@@')
            self._add_gnd_truth(self.search_tree, true_nn_map, global_search_params)
            pop_method = BFS_ORDERING
        else:
            pop_method = None
        return pop_method

    def _add_pseudo_gnd_truth_at_end(self, true_nn_map, global_search_params):
        if FLAGS.add_gnd_truth == ADD_GND_TRUTH_AT_END and self.is_train and (true_nn_map is not None and len(true_nn_map) != 0):
            # print('@@@@@@@@@******')
            self._add_gnd_truth(self.search_tree, true_nn_map, global_search_params)
        if FLAGS.mark_best_as_true:
            self.search_tree.mark_best_as_true(global_search_params.gq.nx_graph.number_of_nodes())

    def _add_gnd_truth(self, search_tree, true_nn_map, global_search_params):
        # print('Here!!!!!!!!')
        true_nn_map = deepcopy(true_nn_map)
        cur_node, cur_nn_map, len_true_nn_map = \
            search_tree.root, {}, len(true_nn_map)
        while len(true_nn_map) > 0:
            # pop from search_tree
            if len(cur_nn_map) == 0:
                gq, query_tree = global_search_params.gq.nx_graph, global_search_params.CS[1]
                u, = get_u_next_li(gq, query_tree, cur_nn_map)
            else:
                u = cur_node.u
            v = true_nn_map.pop(u)

            cur_node.remove_uv_pair(u,v)
            P_idx = cur_node.v_li.index(v)
            assert False, 'above is slow! .index()'
            next_node, _ = self.execute_action(cur_node, global_search_params, u, v, P_idx) # TODO: deprecated?

            self.search_tree.update_search_stack(self, cur_node, next_node, BFS_ORDERING)
            cur_node = next_node

    def get_is_early_pruning(self, cur_node, vscore, global_search_params):
        if FLAGS.use_is_early_pruning:
            assert not global_search_params.MCTS
            reward_accum_pred = \
                get_accum_reward_from_start(
                    len(cur_node.nn_map),
                    self.search_tree.root.gq.nx_graph.number_of_nodes(),
                    cur_node.is_dead_end
                ) + vscore
            is_early_pruning = reward_accum_pred < self.best_reward_accum
        else:
            is_early_pruning = False
        return is_early_pruning

    def execute_action(self, cur_node, global_search_params, u, v, P_idx, gid=None, logger=None,
                       additional_edge_attributes=None, additional_node_attributes=None):
        '''
        This function is called by both outer and inner (MCTS). A tricky and critical function!
        '''
        # find existing nodes
        action2node = \
            {
                self.search_tree.nxgraph.edges[eid]['action']: self.search_tree.nid2node[eid[1]]
                for eid in self.search_tree.nxgraph.out_edges(cur_node.nid)
            }

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear('action2node')

        # check if node already exists
        if (u, v) in action2node:
            # An old node that has been visited by either outer or inner search.
            next_node = action2node[(u, v)]
            is_already_exists = True
            # Need to update the Q, U, etc. for the edge regardless of new node or not.
            if additional_edge_attributes is not None:
                self.search_tree.set_edge_attrib(cur_node, next_node, additional_edge_attributes)
            if FLAGS.time_analysis:
                self.debugger_timer.time_and_clear('(u, v) in action2node')
        else:
            next_node = self._get_next_node(cur_node, global_search_params, u, v,)
            if FLAGS.time_analysis:
                self.debugger_timer.time_and_clear('_get_next_node')
            self.search_tree.add_node_wrapper(
                cur_node, u, v, P_idx, next_node, len(self.solutions), self,
                global_search_params.graph_filter, additional_edge_attributes,
                cs_map=global_search_params.CS[0]
            )
            is_already_exists = False
            if FLAGS.time_analysis:
                self.debugger_timer.time_and_clear('search_tree.add_node_wrapper')
        if additional_node_attributes is not None:
            self.search_tree.set_node_attrib(cur_node, additional_node_attributes)
        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear('search_tree.set_node_attrib')
        return next_node, is_already_exists

    def sample_uv_via_pi(self, cur_node, v_pi_logits_li):
        v_li_valid, v_indices, v_pi_logits_li_valid = cur_node.get_valid_uv_pair_li(v_pi_logits_li)
        # NOTE: tricky code! there are 2 indices here! v_id is the indices in
        # the RCS v_li! v_indices is the indices in the whole v_li!
        v_pi_li_valid = logits2pi(v_pi_logits_li_valid)
        v_id = np.random.choice(len(v_pi_li_valid), p=v_pi_li_valid/sum(v_pi_li_valid))
        u, v, vscore, P_idx = cur_node.u, v_li_valid[v_id], v_pi_li_valid[v_id], v_indices[v_id]
        return u, v, vscore, v_pi_li_valid, P_idx

    def _get_MCTS_num_iters(self, cur_node):
        if len(cur_node.v_li) == 1:
            MCTS_num_iters = 1
        else:
            MCTS_num_iters = \
                min(
                    int(FLAGS.MCTS_num_iters_per_action*len(cur_node.v_li_valid)),
                    FLAGS.MCTS_num_iters_max
                )
        return MCTS_num_iters

    def _run_single_iteration(self, cur_node, global_search_params):

        if global_search_params.MCTS:
            # TODO: did we ensure cur_node is NOT terminal?
            v_pi_logits_li = self._run_MCTS(cur_node, global_search_params, self._get_MCTS_num_iters(cur_node)) # many iters of MC
            self.MCTS_iter += 1
            if v_pi_logits_li is None:
                # Time/Iter out etc.
                return None
            u, v, _, pi_sm_normalized_valid, P_idx = self.sample_uv_via_pi(cur_node, v_pi_logits_li)
            node_attrib = {'pi_sm_normalized_valid': pi_sm_normalized_valid}
            cur_node.remove_uv_pair(u,v)
            if cur_node.v_pi_logits_li is None:
                cur_node.v_pi_logits_li = v_pi_logits_li
        else:
            # Run either DAF/GraphQL or NN to get P(s,a) for every a in the new node.
            u, v, _, P_idx = cur_node.get_next_uv_pair()
            node_attrib = None

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear('MCTS done')

        next_node, is_already_exists = \
            self.execute_action(
                cur_node, global_search_params, u, v, P_idx,
                additional_node_attributes=node_attrib
            )

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear('execute_action')

        # The following code is very tricky! execute_action() creates a new node with N and W being 0.
        # Ideally, inner search (MCTS) handles incrementing N and W, but
        # outer search must visit a "new" node (in terms of outer) EVERY TIME.
        # (Outer and inner work independently -- even without inner, outer can search on its own.)
        # (But then this means we have to do something special when outer creates a new node.)
        # Thus, inner search may see a node created by outer with N and W being 0.
        # Thus, we have to do something special:
        if not is_already_exists and global_search_params.MCTS:
            self._backup_reward(next_node, cur_node) # its possible MCTS never visits v_li_valid => we create a new node (initialize W=out_value, N=1)
        # # is_already_exists can be False with MCTS (ex. only one already selected action explored by MCTS)
        # assert is_already_exists == FLAGS.MCTS, 'We assume MCTS expands cur_node; only when not using MCTS we expand using raw P'
        assert (not is_already_exists) or global_search_params.MCTS, 'should always be a new node if not using MCTS!'

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear('_backup_reward')

        return u, v, next_node

    ############################################# Start of MCTS ######################################################

    def _run_MCTS(self, pseudo_root, global_search_params, MCTS_num_iters):
        # Need to update num_iters, look at timer, etc.
        for i in range(MCTS_num_iters):

            if FLAGS.time_analysis:
                self.debugger_timer.time_and_clear(f'{i} MCTS: begin')

            # Select.
            # print(f'iteration: {i}/{FLAGS.MCTS_num_iters}')
            cur_node = pseudo_root
            num_tree_nodes_before = self.search_tree.get_num_nodes()
            j = 0
            while True:
                if not self._check_exit_conditions(check_stackheap=False):
                    # Need to return now.
                    return None

                # self._log_nn_map_time_iter_tuple(next_node) # TODO: is this needed?
                self.num_iters += 1 # TODO: a separate iter count for MCTS? keep revisiting earlier nodes --> count as iter too?

                if cur_node.is_dead_end:
                    # An existing terminal node is reached.
                    # If MCTS keeps reaching the same terminal node,
                    # then need to tune Q+U to allow more exploration,
                    assert self.search_tree.get_num_nodes() == num_tree_nodes_before
                    # vs = \
                    #     get_reward_immediate(
                    #         len(cur_node.nn_map), len(cur_node.nn_map) - 1,
                    #         self.search_tree.root.gq.nx_graph.number_of_nodes()
                    #     ) # TODO: rename max_depth to nn_map_len_term
                    # print(f'revisitied dead end! remaining_actions={len(cur_node.v_li_valid) > 0}-{len(cur_node.v_li_valid)} depth={len(cur_node.nn_map)}')
                    break

                if FLAGS.time_analysis:
                    self.debugger_timer.time_and_clear(f'{i} {j} MCTS: cur_node.is_dead_end')

                u, v, P_idx, best_QUP = self._select_Q_plus_U(cur_node)

                if FLAGS.time_analysis:
                    self.debugger_timer.time_and_clear(f'{i} {j} MCTS: cur_node._select_Q_plus_U')

                cur_node, is_already_exists = \
                    self.execute_action(
                        cur_node, global_search_params, u, v, P_idx,
                        additional_edge_attributes=best_QUP
                    )

                if FLAGS.time_analysis:
                    self.debugger_timer.time_and_clear(f'{i} {j} MCTS: execute_action')

                if not is_already_exists:
                    # A new node/state should have been added to the tree.
                    assert self.search_tree.get_num_nodes() == num_tree_nodes_before + 1
                    # vs = cur_node.value
                    # print(f'created a new search node! depth={len(cur_node.nn_map)}')
                    break

                if FLAGS.time_analysis:
                    self.debugger_timer.time_and_clear(f'{i} {j} MCTS: is_already_exists')

                j += 1
            # print('tree size:',self.search_tree.nxgraph.number_of_nodes()
            self._backup_reward(cur_node, pseudo_root) # TODO: back up to real root in our case!!!!! turn it into a FLAG

            if FLAGS.time_analysis:
                self.debugger_timer.time_and_clear(f'{i} MCTS: _backup_reward')
                self.debugger_timer.print_durations_log()



        # Compute pi using visiting counts.
        v2node = self.search_tree.get_all_actions_as_v_to_node_dict(pseudo_root)
        pi_logits = [v2node[v][0].N if v in v2node else 0 for v in pseudo_root.v_li]
        # counts = [x ** (1. / FLAGS.MCTS_temp) for x in counts]
        # counts_sum = float(sum(counts))
        # pi = [x / counts_sum for x in counts]
        return pi_logits

    def _select_Q_plus_U(self, cur_node):
        # best_QUP = None
        # best_v = None
        # best_ucb = -float('inf')
        v2node_idx = self.search_tree.get_all_actions_as_v_to_node_dict(cur_node)

        W_child, N_child, indices = [], [] ,[]
        for v, (node, idx) in v2node_idx.items():
            W_child.append(node.W)
            N_child.append(node.N)
            indices.append(idx)
        assert len(W_child) == len(N_child) == len(indices)
        assert len(cur_node.v_li) >= len(W_child)
        W_child, N_child = torch.tensor(W_child, dtype=torch.float32), torch.tensor(N_child, dtype=torch.float32)

        Q = \
            (
                cur_node.out_value - \
                get_reward_accumulated(
                    len(cur_node.nn_map)+1,
                    len(cur_node.nn_map),
                    self.search_tree.root.gq.nx_graph.number_of_nodes(),
                    False
                )
            ) * torch.ones(len(cur_node.v_li))
        Q[indices] = W_child / N_child  # |selected_actions| x 1
        Q = Q # |v_li| x 1

        N_child = \
            scatter_add(
                N_child,
                torch.tensor(indices, dtype=torch.long),
                dim_size=len(cur_node.v_li)
            )
        U = FLAGS.MCTS_cpuct * cur_node.P_li * math.sqrt(cur_node.N + FLAGS.MCTS_eps_in_U) / (1 + N_child) # |v_li| x 1

        UCB = Q + U
        P_idx = int(torch.argmax(UCB).numpy())
        best_ucb = float(UCB[P_idx].numpy())
        best_v = cur_node.v_li[P_idx]
        best_QUP = \
            {
                'best_ucb': best_ucb,
                'Q': float(Q[P_idx].numpy()), 'U': float(U[P_idx].numpy()),
                'best_P': float(cur_node.P_li[P_idx])
            }

        if FLAGS.MCTS_printQU:
            self.MCTS_Qs.extend(Q.tolist())
            self.MCTS_Us.extend(U.tolist())
            self.MCTS_QU.extend((Q >= U).float().tolist())

        # for i, a in enumerate(cur_node.v_li): # action space v_li should not be modified by outer search
        #     child = v2node.get(a)
        #     assert len(cur_node.P_li) > 0, 'Must call NN predict or DAF random init for P_li before this line!'
        #     if child is not None:
        #         # Q = V(st+1)
        #         Q = child.W / child.N # technically NOT Q(s,a) --> instead, it is V(s_{s+1})
        #         U = FLAGS.MCTS_cpuct * cur_node.P_li[i] * math.sqrt(cur_node.N) / (1 + child.N)
        #     else:
        #         # Very tricky code! In AlphaGo, r = 0. But in our case, we have to consider r.
        #         # Vs = max Q(s,a) = max(r + Vst+1) = r + Vst+1 => Vst+1 = Vs - r
        #         # V(st+1) = V(st) - r(st,st+1)
        #         Q = \
        #             cur_node.out_value - \
        #             get_reward_accumulated(
        #                 len(cur_node.nn_map)+1,
        #                 len(cur_node.nn_map),
        #                 self.search_tree.root.gq.nx_graph.number_of_nodes(),
        #                 False
        #             )
        #         U = FLAGS.MCTS_cpuct * cur_node.P_li[i] * math.sqrt(cur_node.N + FLAGS.MCTS_eps_in_U)  # Q = 0 ?
        #     # try:
        #     if FLAGS.MCTS_printQU:
        #         self.MCTS_Qs.append(Q)
        #         self.MCTS_Us.append(U)
        #         self.MCTS_QU.append(1 if Q >= U else 0)
        #     if Q+U >= best_ucb: # TODO: > or >=?
        #         best_v = a
        #         best_ucb = Q+U
        #         best_QUP = \
        #             {
        #                 'best_ucb':float(best_ucb),
        #                 'Q':float(Q), 'U':float(U),
        #                 'best_P':float(cur_node.P_li[i])
        #             }

        # print(f'@@@@@ cur_node.u={cur_node.u} best_v={best_v} out of {len(cur_node.v_li)} best_ucb={best_ucb}')
        assert best_v is not None
        return cur_node.u, best_v, P_idx, best_QUP


    def _backup_reward(self, leaf_node, pseudo_root):
        assert leaf_node != pseudo_root, 'Something is wrong! leaf node is equal to pseudo root and nothing to back up'

        # if float(leaf_node.out_value) != 0:
        #     print(f'leaf_node.out_value before', leaf_node.out_value)
        if leaf_node.is_dead_end:
            v = 0.0
        else:
            v = leaf_node.out_value # TODO get predicted v by NN somehow
            assert v is not None, 'value of leaf_node is None!'
        cur_node = leaf_node
        while True:
            # if float(leaf_node.out_value) != 0:
            #     print(f'leaf_node.out_value middle', leaf_node.out_value)
            assert len(leaf_node.nn_map) >= len(cur_node.nn_map)
            delta = \
                get_reward_accumulated(
                    len(leaf_node.nn_map),
                    len(cur_node.nn_map),
                    self.search_tree.root.gq.nx_graph.number_of_nodes(),
                    leaf_node.is_dead_end
                )
            assert delta >= 0
            # if float(leaf_node.out_value) != 0:
            #     print(f'leaf_node.out_value middle 2', leaf_node.out_value)
            cur_node.W += (delta + v)
            # tmp = delta + v
            # cur_node.W += tmp


            # if float(leaf_node.out_value) != 0:
            #     print(f'leaf_node.out_value middle 3', leaf_node.out_value)
            cur_node.N += 1
            self.search_tree.nxgraph.nodes[cur_node.nid]['W'] = str(float(cur_node.W))
            self.search_tree.nxgraph.nodes[cur_node.nid]['N'] = cur_node.N
            # if float(leaf_node.out_value) != 0:
            #     print(f'leaf_node.out_value middle 4', leaf_node.out_value)
            if FLAGS.MCTS_backup_to_real_root:
                if cur_node == self.search_tree.root:
                    break
            else:
                if cur_node == pseudo_root:
                    break
            cur_node = cur_node.parent
        # if float(leaf_node.out_value) != 0:
        #     print(f'leaf_node.out_value after', leaf_node.out_value)

    ############################################# End of MCTS ######################################################
    def _get_next_node(self, cur_node, global_search_params, u, v):
        # update nn_map
        nn_map_new = self.add_pair_to_nn_map(cur_node.nn_map, u, v)
        filter_key = global_search_params.graph_filter.check_if_new_filter(cur_node, nn_map_new)

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear('check_if_new_filter')

        # run neural network
        out, candidate_map = self._get_next_u_and_vli(global_search_params, nn_map_new, filter_key)
        if out is None:
            # terminal node
            u_next, v_li_next, out_policy, out_value, out_other = None, [], [], 0.0, None
            is_dead_end = True
        else:
            # non-terminal new node
            u_next, v_li_next, out_policy, out_value, out_other = out
            is_dead_end = False

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear('run neural network')

        # create the new node
        gq, gt = global_search_params.gq, global_search_params.gt
        next_node = \
            SearchNode(
                cur_node, u_next, v_li_next, out_policy, out_value, out_other,
                nn_map_new, gq, gt, candidate_map, filter_key, is_dead_end,
                self.get_is_early_pruning(cur_node, out_value, global_search_params)
            )

        # update the incumbent
        self._update_best_solution(next_node, global_search_params.gid, global_search_params.logger)

        # update the solution set
        is_solution = self._check_is_solution(next_node.nn_map, global_search_params.gq)
        if is_solution:
            # exit(-1)
            self.solutions.append(deepcopy(next_node.nn_map))
            if self.is_train:
                self.search_tree.mark_solution(next_node, frozenset(next_node.nn_map.values()))

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear('update')

        return next_node

    #######################################################
    def _get_next_u_and_vli(self, global_search_params, nn_map, filter_key):
        '''
        returning None means the state is a dead end!!
        '''

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear(f'get_candidate_map before {random.random()}')


        candidate_map = self.get_candidate_map(global_search_params, nn_map)

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear(f'get_candidate_map done {random.random()}')

        if FLAGS.use_NN_for_u_score: # TODO: flag gone?????
            assert False # do we handle V from P_li, V correctly?
            out = self.compute_u_v_score_iterative(global_search_params, nn_map, candidate_map, filter_key)
        else:
            out = self.compute_u_v_score_wrapper(global_search_params, nn_map, candidate_map, filter_key)
        return out, candidate_map

    def compute_u_v_score_wrapper(self, global_search_params, nn_map, candidate_map, filter_key):
        '''
        returning None means the state is a dead end!!
        '''
        if len(candidate_map) == 0:
            return None  # dead end! no suitable (u,v) pairs from bidomain pruning

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear(f'start of compute_u_v_score_wrapper {random.random()}')

        gq, gt, CS, daf_path_weights, model, graph_filter, override_use_heuristic, eps, align_matrix = \
            global_search_params.get_params()
        cs_map, query_tree = CS

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear(f'global_search_params.get_params {random.random()}')

        # print('@@@ override_use_heuristic', override_use_heuristic)
        # get the "best u"
        u_score_li, u_li = self.get_u_score_li(candidate_map, daf_path_weights)

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear(f'get_u_score_li u={len(u_li)}')

        u_score, u_best = min(list(zip(u_score_li, u_li)), key=lambda x: x[0]) # TODO: check! smaller is better

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear(f'min(list(zip(u_score_li, u_li)) u={u_best}')

            # self.debugger_timer.print_durations_log()
            # exit(-1)

        # saver.log_info(f'u_li: {u_li}')
        # saver.log_info(f'u_score_li: {u_score_li}')
        # saver.log_info(f'u_best {u_best}: {len(candidate_map[u_best])}')
        # saver.log_info(f'neighbors of 3: {list(nx.neighbors(gq.nx_graph, 3))}')
        # saver.log_info(f'neighbors of 8: {list(nx.neighbors(gq.nx_graph, 8))}')
        # saver.log_info(f'neighbors: {list(nx.neighbors(gq.nx_graph, u_best))}')

        # rank v_li = u2v_li["best u"]
        v_li = candidate_map[u_best]
        out_policy, out_value, out_other = \
            self.get_v_score_li(
                u_best, v_li, model, graph_filter, filter_key, gq, gt,
                nn_map, cs_map, candidate_map, override_use_heuristic,
                eps=eps, align_matrix=align_matrix, query_tree=query_tree
            )

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear(f'get_v_score_li slow? {len(v_li)} u={u_best}')

        return u_best, v_li, out_policy, out_value, out_other

    def compute_u_v_score_iterative(self, global_search_params, nn_map, candidate_map, filter_key):
        '''
        returning None means the state is a dead end!!
        '''
        if len(candidate_map) == 0:
            return None # dead end! no suitable (u,v) pairs from bidomain pruning

        gq, gt, CS, daf_path_weights, model, graph_filter, override_use_heuristic, eps, align_matrix = \
            global_search_params.get_params()
        cs_map, query_tree = CS

        u_score_li, u_li, u2v_score_li = [], [], {}
        AGG = lambda x: x.max().detach().cpu().numpy()

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear('harmless stuff')

        for u, v_li in candidate_map.items():
            out_policy, out_value, _ = \
                self.get_v_score_li(
                    u, v_li, model, graph_filter, filter_key, gq, gt,
                    nn_map, cs_map, candidate_map, override_use_heuristic,
                    eps=eps, require_abs_score=True, align_matrix=align_matrix,
                    query_tree=None
                )

            u_score_li.append(AGG(out_policy))
            u_li.append(u)

            u2v_score_li[u] = (v_li, out_policy, out_value)

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear(f'slow? {len(candidate_map)}')

        u_score, u_best = max(list(zip(u_score_li, u_li)), key=lambda x: x[0])
        v_li, out_policy, out_value = u2v_score_li[u_best]
        return u_best, v_li, out_policy, out_value, None

    def get_candidate_map(self, global_search_params, nn_map):
        gq, gt, CS, daf_path_weights, model, _, override_use_heuristic, _, _ = \
            global_search_params.get_params()
        cs_map, query_tree = CS
        gq, gt = gq.nx_graph, gt.nx_graph

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear(f'get_candidate_map start {random.random()}')


        u_next_li = get_u_next_li(gq, query_tree, nn_map)

        # if FLAGS.time_analysis:
        #     self.debugger_timer.time_and_clear(f'get_u_next_li {random.random()}')


        candidate_map = {}
        for u in sorted(u_next_li):
            # if FLAGS.time_analysis:
            #     self.debugger_timer.time_and_clear(f'for loop start {random.random()} {u}')
            v_candidates, is_root = get_v_candidates_wrapper(u, nn_map, gq, gt, cs_map, query_tree)

            # if FLAGS.time_analysis:
            #     self.debugger_timer.time_and_clear(f'get_v_candidates_wrapper {random.random()} {u}')


            if len(v_candidates) > 0:
                candidate_map[u] = v_candidates
            else:
                return {} # NOTE: tricky bug! We should not continune search! If any u node has an empty v_li then we should immeidately backtrack!
            assert not is_root or len(u_next_li) == 1

            if global_search_params.logger is not None and self.is_train:
                global_search_params.logger.update_cs_bd_intersection(
                    global_search_params.logger.search_num, cs_map[u], v_candidates)

            # if FLAGS.time_analysis:
            #     self.debugger_timer.time_and_clear(f'update_cs_bd_intersection {random.random()} {u}')
        assert len(candidate_map) == len(u_next_li)
        return candidate_map

    def get_u_score_li(self, u2v_candidates, daf_path_weights):
        u_score_li, u_li = [], []
        for u, v_candidates in u2v_candidates.items():
            if FLAGS.matching_order == 'DAF':
                u_score = sum([daf_path_weights[u][v] for v in v_candidates])
            elif FLAGS.matching_order in ['GraphQL', 'nn']:
                u_score = len(v_candidates)
            else:
                assert False
            u_score_li.append(u_score)
            u_li.append(u)
        return u_score_li, u_li

    def get_v_score_li(self, u, v_li, model, graph_filter, filter_key,
                       gq, gt, nn_map, cs_map, candidate_map,
                       override_use_heuristic, eps=0.0,
                       require_abs_score=False, align_matrix=None,
                       query_tree=None):
        cond1 = not override_use_heuristic
        cond2 = len(v_li) > 1 or require_abs_score
        cond3 = random.random() >= eps
        if FLAGS.skip_if_action_space_less_than is None:
            cond4 = True
        else:
            cond4 = len(v_li) >= FLAGS.skip_if_action_space_less_than

        if FLAGS.time_analysis:
            self.debugger_timer.time_and_clear(f'cond {len(v_li)} u={u}')

        # if not cond4:
        #     print(f'skip nn beacuse action_size = {len(v_li)} < {FLAGS.skip_if_action_space_less_than}')
        # print(cond1, cond2, cond3)
        if align_matrix is not None:
            out_policy = gen_alignment_matrix(align_matrix, gq.nx_graph, gt.nx_graph, u, v_li, self.timer, FLAGS.timeout, method_type="order")
            out_value = 0.0
            out_other = None
        elif FLAGS.matching_order == 'nn' \
                and cond1 \
                and cond2 \
                and cond3 \
                and cond4:
            out_policy, out_value, out_other = \
                model(
                    gq, gt, u, v_li, nn_map, cs_map, candidate_map,
                    FLAGS.cache_embeddings, graph_filter=graph_filter,
                    filter_key=filter_key, execute_action=self.get_candidate_map,
                    query_tree=query_tree
                )

            if FLAGS.time_analysis:
                self.debugger_timer.time_and_clear(f'slow??? model {len(v_li)} u={u}')

            out_policy = out_policy.cpu().detach().numpy().tolist()
            out_value = out_value.cpu().detach().item()

            # if max(out_value) > 1000:
            #     print('#####')
            #     exit()
            self.nn_call_cnt += 1
        else:
            out_policy = np.zeros(len(v_li)) # TODO: set all zeros or random?
            out_value = out_policy[0]
            out_other = None
            self.eps_call_cnt += 1
            if cond1:
                self.eps_cond1_call_cnt += 1
            if cond2:
                self.eps_cond2_call_cnt += 1
            if cond3:
                self.eps_cond3_call_cnt += 1
            if cond4:
                self.eps_cond4_call_cnt += 1
        return out_policy, out_value, out_other

    def _create_root(self, global_search_params, nn_map):
        filter_key = None
        out, candidate_map = self._get_next_u_and_vli(global_search_params, nn_map, filter_key)

        is_not_a_subgraph = False
        for v_li in global_search_params.CS[0].values():
            if len(v_li) == 0:
                is_not_a_subgraph = True
                break

        if out is None or is_not_a_subgraph:
            saver.log_info('WARNING:gq is not a subgraph of gt!!!')
            return None  # gq is not a subgraph of gt!
        else:
            u, v_li, out_policy, out_value, out_other = out
            root = \
                SearchNode(
                    None, u, v_li, out_policy, out_value, out_other, {},
                    global_search_params.gq, global_search_params.gt,
                    candidate_map, filter_key, False, False
                )
            return root

    # def _log_nn_map_time_iter_tuple(self, state):
    #     nn_map_best = len(self.best_nn_map)
    #     nn_map = len(state.nn_map)
    #     time = self.timer.get_duration()
    #     iteration = self.num_iters
    #     # self.search_stream_plot_logger.add_point(nn_map_best, nn_map, time, iteration)

    def _check_exit_conditions(self, check_stackheap=True):
        is_timeout = self._check_timeout()
        is_iterout = self._check_iterout()
        if check_stackheap:
            is_emptysearchtree = self.search_tree.check_empty()
        else:
            is_emptysearchtree = False
        # print('is_timeout', is_timeout)
        # print('is_iterout', is_iterout)
        # print('is_emptysearchtree', is_emptysearchtree)
        return not (is_timeout or is_iterout or is_emptysearchtree)

    def add_pair_to_nn_map(self, nn_map, u, v):
        new_nn_map = deepcopy(nn_map)
        new_nn_map[u] = v
        return new_nn_map

    def _check_is_solution(self, nn_map, gq):
        return len(nn_map) == gq.nx_graph.number_of_nodes()

    #######################################################

    def _check_need_return(self, score, scores):
        # print('not self.is_train', not self.is_train)
        # print('FLAGS.Qtrick', FLAGS.Qtrick)
        # print(f'score {score} < 0.5 * max(scores) {0.5 * max(scores)}', score < 0.5 * max(scores))
        if not self.is_train and FLAGS.Qtrick and score < 0.5 * max(scores):
            saver.log_info(f'anonymous Q trick')
            return True
        if self.num_iters == self.num_iters_threshold:
            saver.log_info(f'Iterations used up (2)! {self.num_iters}')
            return True
        return False

    #######################################################

    def _check_timeout(self):
        seconds = self.timer.get_duration()
        is_timeout = FLAGS.timeout != -1 and seconds > FLAGS.timeout
        if is_timeout:
            saver.log_info(f'Time up! {format_seconds(seconds)}')
        return is_timeout

    def _check_iterout(self):
        is_iterout = self.num_iters_threshold != -1 and self.num_iters == self.num_iters_threshold
        if is_iterout:
            saver.log_info(f'Iterations used up (1)! {self.num_iters} iters')
            # exit()
        return is_iterout

    def _check_found_solution(self, nn_map, gq):
        return len(nn_map) == gq.nx_graph.number_of_nodes()

    def _update_best_solution(self, next_node, gid, logger):
        nn_map = next_node.nn_map
        if len(nn_map) > len(self.best_nn_map):  # could be multiple solutions; record the first
            self.best_nn_map = deepcopy(nn_map)
            self.best_nn_map_iter = self.num_iters
            if logger is not None and not self.is_train:
                logger.gid2iteration_time_best_nn_map[gid].append({
                    'iteration':self.num_iters,
                    'time':self.timer.get_duration(),
                    'best_nn_map':self.best_nn_map,
                    'best_nn_map_size':len(self.best_nn_map)
                })
        if FLAGS.use_is_early_pruning:
            reward_accum = \
                get_accum_reward_from_start(
                    len(next_node.nn_map),
                    self.search_tree.root.gq.nx_graph.number_of_nodes(),
                    next_node.is_dead_end
                )
            self.best_reward_accum = max(self.best_reward_accum, reward_accum)


# USE_HEURISTIC = True

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def eval_print_search_result(smts, gq):
    num_solutions = len(smts.solutions)
    num_solutions_unique = len(_get_unique_node_solutions(smts.solutions))

    best_size = len(smts.best_nn_map)
    best_ratio = best_size / gq.nx_graph.number_of_nodes()

    saver.log_info(
        f'num_solutions: {num_solutions}; num_solutions_unique: {num_solutions_unique}')
    saver.log_info(f'best_size: {best_size} nodes at iter {smts.best_nn_map_iter} '
                   f'(/{gq.nx_graph.number_of_nodes()}={best_ratio:.4f});'
                   f' total iters: {smts.num_iters} (smts.MCTS_iter={smts.MCTS_iter})')
    saver.log_info(
        f'# NN calls: {smts.nn_call_cnt}; # eps "calls": {smts.eps_call_cnt}; # cond1 "calls": {smts.eps_cond1_call_cnt}; # cond2 "calls": {smts.eps_cond2_call_cnt}; # cond3 "calls": {smts.eps_cond3_call_cnt}; # cond4 "calls": {smts.eps_cond4_call_cnt}')
    if FLAGS.MCTS_printQU:
        print_stats(smts.MCTS_Qs, 'MCTS_Qs', print_func=saver.log_info)
        print_stats(smts.MCTS_Us, 'MCTS_Us', print_func=saver.log_info)
        print_stats(smts.MCTS_QU, 'MCTS_QU(Q winning over U)', print_func=saver.log_info)
    return num_solutions, num_solutions_unique, best_size, best_ratio


def _get_unique_node_solutions(solutions):
    rtn = set()
    for sol in solutions:
        node_set = frozenset(sol.values())
        rtn.add(node_set)
    return rtn
