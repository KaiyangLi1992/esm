from NSUBS.src.utils import OurTimer
from NSUBS.model.OurSGM.saver import saver

import networkx as nx
import numpy as np
import math
import itertools

from copy import deepcopy
from NSUBS.model.OurSGM.data_structures_common import QsaObj, StackHeap
from NSUBS.model.OurSGM.dvn_wrapper import create_u2v_li, GraphFilter
from NSUBS.model.OurSGM.config import FLAGS, AREA, POSITIVE_PAIRS, COVERAGE, BINARY, DEPTH, EXP_DEPTH, EXP_DEPTH_RAW

from NSUBS.model.OurSGM.utils_our import get_reward_accumulated, get_accum_reward_from_start

BFS_ORDERING = 'bfs_ordering'
SCORE_ORDERING = 'score_ordering'
PROBABILISTIC_SCORE_ORDERING = 'probabilistic_score_ordering'

WEIGHT_CRITERION_SEARCH_DEPTH = 1
WEIGHT_CRITERION_NUM_STATES = 0.5
WEIGHT_CRITERION_BIAS = 0.0#1#1

###################################################################################
class RegretTrigger():
    def __init__(self, thresh, backtrack_ordering):
        self.thresh = thresh
        self.local_best = 0
        self.iters_since_local_best_updated = 0
        self.backtrack_ordering = backtrack_ordering

    def trigger(self, next_state, pop_method):
        if pop_method != BFS_ORDERING:
            self._reset(next_state)
        else:
            is_local_best_updated = len(next_state.nn_map) > self.local_best
            self._update_iters_since_local_best_updated(is_local_best_updated, next_state)

        if self.thresh == -1 \
                or self.iters_since_local_best_updated <= self.thresh \
                or not next_state.is_leaf():
            return BFS_ORDERING
        else:
            return self.backtrack_ordering

    def _reset(self, state):
        self.local_best = len(state.nn_map)
        self.iters_since_local_best_updated = 0

    def _update_iters_since_local_best_updated(self, is_local_best_updated, next_state):
        if is_local_best_updated:
            self.local_best = len(next_state.nn_map)
            self.iters_since_local_best_updated = 0
        else:
            self.iters_since_local_best_updated += 1

class SearchTree:
    def __init__(self, root, smts, is_train, cs_map=None):
        self.root = root
        self.score2nid_stackheap = StackHeap()
        self.nid2node = {}
        self.nid = 0
        self.nxgraph = nx.DiGraph()
        # TODO: tune the trigger and criterions
        if is_train:
            self.trigger = RegretTrigger(FLAGS.regret_iters_train, PROBABILISTIC_SCORE_ORDERING)
            self.criterion = self.train_criterion
        else:
            self.trigger = RegretTrigger(FLAGS.regret_iters_test, SCORE_ORDERING)
            self.criterion = self.train_criterion # TEST_CRITERION IS BAD!

        self.add_node_to_search_tree(root, smts, None, cs_map=cs_map)
        self._update_heapq(root)

    def check_empty(self):
        return len(self.score2nid_stackheap) == 0

    def get_num_nodes(self): # TODO: anonymous: Chexk this implementation!!!!!! Needed by MCTS to do some assertion check!
        return self.nxgraph.number_of_nodes()

    def get_all_actions_as_v_to_node_dict(self, node): # TODO: anonymous: Chexk this implementation!!!!!!
        edges = self.nxgraph.out_edges(node.nid)
        uv_node = {self.nxgraph.edges[edge]['action'][1]: (self.nid2node[edge[1]], self.nxgraph.edges[edge]['idx']) for edge in edges}
        return uv_node

    def train_criterion(self, node):
        return WEIGHT_CRITERION_SEARCH_DEPTH * node.search_depth() + \
               WEIGHT_CRITERION_NUM_STATES * (node.number_of_states_remaining_to_explore() / len(node.v_li)) + \
               WEIGHT_CRITERION_BIAS * (node.out_value + get_accum_reward_from_start(len(node.nn_map), node.gq.nx_graph.number_of_nodes(), False))
        # return min(0.4, self.test_criterion(node) / 1000) + 0.6 * random.random()

    def test_criterion(self, node):
        return len(node.sorted_v_li_indices)  # TODO: diff criterion for heapq?

    def fill_depths(self, node):
        depth = 0 if node.parent is None else node.parent.depth + 1

        node.depth = depth
        node.max_depth = depth
        node = node.parent
        while node is not None:
            if depth >= node.max_depth:
                node.max_depth = depth
                nx.set_node_attributes(
                   self.nxgraph,
                   {node.nid: {'max_depth': depth}}
                )
                node = node.parent
            else:
                break
        return depth

    def get_circle_size(self, node, graph_filter, cs_map):
        if cs_map is None: # terminal node
            circle_size = -1
        elif len(node.nn_map) == 0: # root node
            assert len(set(itertools.chain.from_iterable([v_li for v_li in cs_map.values()]))) == node.gt.nx_graph.number_of_nodes()
            circle_size = node.gt.nx_graph.number_of_nodes()
        else:
            u2v_li = create_u2v_li(node.nn_map, cs_map, node.candidate_map)
            _, node_mask_inv = graph_filter.get_node_mask(node.filter_key, node.gq.nx_graph, node.gt.nx_graph, node.nn_map.values(), u2v_li)
            circle_size = len(node_mask_inv)
        return circle_size

    def add_node_to_search_tree(self, node, smts, graph_filter, cs_map=None):
        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()

        action_space_size = len(node.candidate_map[node.u]) if node.u in node.candidate_map else 0
        assert len(node.v_li) == action_space_size
        self.nid2node[self.nid] = node
        node.register_nid(self.nid)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'quick')

        depth = self.fill_depths(node)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'fill_depths')

        # if type(node.out_value) is not np.float64:
        #     node.out_value = list(node.out_value)
        # node.out_value = float(node.out_value)

        # print('nid@@', node.nid, node.out_value)
        # if 2 in self.nxgraph.nodes:
        #     print(self.nxgraph.nodes[2]['out_value'])
        # if float(node.out_value) > 1000:
        #     print('!!!!!!!', float(node.out_value))
        #
        #     exit()

        # G = self.nxgraph
        # for nid in G.nodes:
        #     if float(G.nodes[nid]['out_value']) > 1000:
        #         print('????????', float(G.nodes[nid]['out_value']))
        #         print(G.nodes[nid]['out_value'])
        #         print('nid is', nid)
        #         exit()

        circle = None#self.get_circle_size(node, graph_filter, cs_map)
        if FLAGS.time_analysis:
            timer.time_and_clear(f'get_circle_size')


        self.nxgraph.add_node(
            self.nid, N=0, W=0,            u=node.u,
            vli_size=len(node.v_li),
            depth=depth, search_stage='inner',
            action_space_size=action_space_size,
            circle_size=circle,
            nid=self.nid,
            cur_best_size=len(smts.best_nn_map),
            cur_best_iter=smts.best_nn_map_iter,
            num_iters=smts.num_iters,
            MCTS_iter=smts.MCTS_iter,
            nn_call_cnt=smts.nn_call_cnt,
            eps_call_cnt=smts.eps_call_cnt,
            solutions=len(smts.solutions),
            out_policy=node.out_policy,
            out_value=node.out_value,
            pi_sm_normalized_valid=[],
            max_depth=node.max_depth,
        )#len(node.nn_map))  # , depth=node.depth, num_solutions=node.num_solutions, uv=node.uv)
        self.nid += 1

        if FLAGS.time_analysis:
            timer.print_durations_log()

    def get_node(self, pop_method):
        if pop_method is None:
            assert len(self.nid2node) == 1
            assert next(iter(self.nid2node.keys())) == self.root.nid
            assert len(self.score2nid_stackheap) == 1
            nid, _ = self.score2nid_stackheap.pop_task('stack')
            next_node = self.nid2node[nid]
        elif pop_method == BFS_ORDERING:
            nid, _ = self.score2nid_stackheap.pop_task('stack')
            next_node = self.nid2node[nid]
        elif pop_method == PROBABILISTIC_SCORE_ORDERING:
            nid, _ = self.score2nid_stackheap.pop_task('prob_heap')
            next_node = self.nid2node[nid]
        elif pop_method == SCORE_ORDERING:
            nid, _ = self.score2nid_stackheap.pop_task('heap')
            next_node = self.nid2node[nid]
        else:
            assert False
        return next_node

    def _update_heapq(self, node):
        if not node.is_dead_end and not node.is_exhausted:
            priority = \
                self.criterion(node) if not node.is_early_pruning \
                    else min(self.score2nid_stackheap.priority_map.values())
            self.score2nid_stackheap.add(node.nid, priority=priority)

    def add_node_wrapper(self, cur_node, u, v, P_idx, next_node, global_num_solutions, smts, graph_filter, additional_edge_attributes, cs_map=None):
        # actually linking the states
        assert cur_node.nid in self.nid2node
        self.add_node_to_search_tree(next_node, smts, graph_filter, cs_map=cs_map)
        nx.set_node_attributes(self.nxgraph,
            {
                next_node.nid: {
                    'u_prime': u,
                    'v_prime': v, 'global_num_solutions': global_num_solutions
                } # TODO: add to cur or next?
            })
        self.nxgraph.add_edge(cur_node.nid, next_node.nid, action=(u, v), idx=P_idx)
        if additional_edge_attributes is not None:
            nx.set_edge_attributes(self.nxgraph, {(cur_node.nid, next_node.nid):additional_edge_attributes}) # link parent --> child (direction matters!)

    def set_edge_attrib(self, cur_node, next_node, edge_attribs):
        nx.set_edge_attributes(self.nxgraph, {(cur_node.nid, next_node.nid): edge_attribs})

    def set_node_attrib(self, cur_node, node_attribs):
        nx.set_node_attributes(self.nxgraph, {cur_node.nid: node_attribs})

    def update_search_stack(self, cur_node, next_node, pop_method):
        # get the next pop method
        self._update_heapq(cur_node)
        self._update_heapq(next_node)
        pop_method = self.trigger.trigger(next_node, pop_method)
        # nx.set_node_attributes(self.nxgraph, {next_node.nid: {'pop_method': pop_method}}) # TODO: add to cur or next?
        return pop_method

    def mark_best_as_true(self, gq_size):
        max_depth_leaves, max_depth = [], -1
        for nid in self.nxgraph.nodes():
            if self.nid2node[nid].is_leaf():
                depth = len(self.nid2node[nid].nn_map)
                if depth > max_depth:
                    max_depth = depth
                    max_depth_leaves = [nid]
                elif depth == max_depth:
                    max_depth_leaves.append(nid)
        if max_depth < gq_size:
            for nid in max_depth_leaves:
                self.mark_solution(self.nid2node[nid], frozenset(self.nid2node[nid].nn_map.values()))

    def mark_solution(self, node, solution):
        while node is not None:
            node.solution_set.add(solution)
            node = node.parent
        # nid = node.nid
        # assert nid in self.nxgraph.nodes() and nid in self.nid2node and nid is not None
        # while nid is not None:
        #     self.nid2node[nid].solution_set.add(solution)
        #     # self.nid2node[nid].ns += 1
        #     N = list(nx.neighbors(self.nxgraph.reverse(), nid))
        #     if len(N) == 0:
        #         assert nid == self.root.nid
        #         nid = None
        #     elif len(N) == 1:
        #         assert len(N) == 1
        #         nid = N[0]
        #     else:
        #         print(f'in search tree, nid {nid} had {len(N)}'
        #               f' parents when it should only have 0 or 1!')
        #         assert False

    def get_children(self, nid):
        return list(nx.neighbors(self.nxgraph, nid))

    def get_u_v_li_executed(self, nid, children):
        u_v_li = [self.nxgraph[nid][child_nid]['action'] for child_nid in children]
        if len(u_v_li) == 0:
            u_li, v_li = [], []
        else:
            u_li, v_li = list(map(list, zip(*u_v_li)))
        assert all(u == u_li[0] for u in u_li)
        assert len(v_li) == len(set(v_li)), f'v_li duplicates {v_li}'
        return u_li[0], v_li

    def get_QsaObj_li(self, global_search_params, graph_filter):
        qsaobj_li = []
        for nid in self.nxgraph.nodes():
            children = self.get_children(nid)  # must be ordered!
            if len(children) > 1:
                # record the Qsa
                r_accum = \
                    get_reward_accumulated(
                        self.nid2node[nid].max_depth,
                        len(self.nid2node[nid].nn_map)-1,
                        self.root.gq.nx_graph.number_of_nodes(),
                        True
                    )
                self.nxgraph.nodes[nid]['z'] = float(r_accum)
                v_r_accum_li_sampled = \
                    [
                        get_reward_accumulated(
                            self.nid2node[child_nid].max_depth,
                            len(self.nid2node[child_nid].nn_map)-1,
                            self.root.gq.nx_graph.number_of_nodes(),
                            True
                        ) for child_nid in children
                    ]
                v_pi_logits_li = self.nid2node[nid].v_pi_logits_li
                if global_search_params.logger is not None:
                    global_search_params.logger.update_emb_logger_value(
                        self.nid2node[nid], float(r_accum),
                        float(get_reward_accumulated(self.nid2node[nid].max_depth, 0, self.root.gq.nx_graph.number_of_nodes(), True))
                    )

                assert v_pi_logits_li is None or len(v_pi_logits_li) > 1 # None means either (1) timed out or (2) inner loop not outer loop
                if (FLAGS.prune_trivial_rewards and self._is_trivial_reward(v_r_accum_li_sampled)) or \
                        (global_search_params.MCTS and v_pi_logits_li is None):
                    continue

                # global_search_params.logger.update_emb_logger_policy(self.nid2node[nid], v_pi_logits_li)

                u, v_li_sampled = self.get_u_v_li_executed(nid, children)
                assert u == self.nid2node[nid].u
                qsaobj_li.append(
                    QsaObj(
                        self.nid2node[nid].u, self.nid2node[nid].v_li, v_li_sampled,
                        r_accum, v_pi_logits_li, v_r_accum_li_sampled,
                        self.nid2node[nid].nn_map, self.nid2node[nid].filter_key,
                        self.nid2node[nid].gq, self.nid2node[nid].gt,
                        global_search_params.CS, self.nid2node[nid].candidate_map,
                        graph_filter
                    )
                )
            else:
                self.nxgraph.nodes[nid]['z'] = 'None'
        return qsaobj_li

    def _is_trivial_reward(self, r_li):
        r_max = max(r_li)
        r_min = min(r_li)
        return (r_max - r_min) <= FLAGS.thresh_throwaway_qsa

    def _compute_max_v_score(self, nid, agg = lambda x: max(x)):
        if self.nid2node[nid].v_score_for_glsearch is None:
            if len(list(nx.neighbors(self.nxgraph, nid))) == 0:
                v_score = 0.0
            else:
                v_score_li = \
                    [
                        self._compute_max_v_score(nidc, agg) \
                        for nidc in nx.neighbors(self.nxgraph, nid)
                    ]
                v_score = 1.0 + agg(v_score_li)
            self.nid2node[nid].v_score_for_glsearch = v_score
        return self.nid2node[nid].v_score_for_glsearch

    def _is_solution_valid(self, solution, nn_map, u, v_li):
        return u is not None and all([solution[u] == nn_map[u] for u in nn_map.keys()]) and solution[u] in v_li

    def get_precision_metrics(self):
        assert False

    def get_QsaObj_li_glsearch(self, global_search_params, graph_filter):
        assert False
        # qsaobj_li = []
        # CS_u2v, _ = global_search_params.CS
        # self._compute_max_v_score(self.root.nid)
        # for nid in self.nxgraph.nodes():
        #     if nid != self.root.nid and self.nid2node[nid].v_score_for_glsearch > 0:
        #         parent_nid, = nx.neighbors(self.nxgraph.reverse(), nid)
        #         u, v = self.nxgraph[parent_nid][nid]['action']
        #         qsaobj_li.append(
        #             QsaObj(
        #                 [self.nid2node[nid].v_score_for_glsearch],
        #                 u, [v], self.nid2node[nid].nn_map,
        #                 self.nid2node[nid].filter_key,
        #                 self.nid2node[nid].gq, self.nid2node[nid].gt, CS_u2v, graph_filter
        #             )
        #         )
        # return qsaobj_li

class SearchNode:
    def __init__(self, parent, u, v_li, out_policy, out_value, out_other,  nn_map, gq, gt, candidate_map, filter_key, is_dead_end, is_early_pruning):
        self.nid = None
        self.u = u
        self.v_li = v_li # entire action space; the visited v_li is in SearchTree's nx_graph
        self.out_policy = out_policy # use policy network to rank actions
        self.candidate_map = candidate_map
        self.nn_map = deepcopy(nn_map) # s0 -> 0,1 -> s1 -> 1,5 -> s2: {0:1, 1:5, 3:4} -> 3,4 -> s3
        self.gq = gq
        self.gt = gt
        self.filter_key = filter_key
        self.is_dead_end = is_dead_end
        self.is_early_pruning = is_early_pruning
        self.parent = parent
        self.depth = None
        self.max_depth = len(nn_map)

        ###### MCTS Variables ######
        self.N = 0


        self.W = 0
        if len(out_policy) > 0:
            P_logits = np.array(out_policy)-np.array(out_policy).max()
            if FLAGS.MCTS_train or FLAGS.MCTS_test:
                P_logits /= FLAGS.MCTS_temp_inner
            self.P_li = np.exp(P_logits)/np.sum(np.exp(P_logits))
            # P_logits = np.array(out_policy)-np.array(out_policy).min()
            # self.P_li = P_logits/np.sum(P_logits)
        else:
            self.P_li = []
        self.out_value = out_value # TODO: must assign something to it! Also what if DAF NOT NN? randomly assign 0 to it?
        if out_value > 1000:
            saver.log_info(f'Warning: Huge out_value {out_value} observed')
            # exit()
        self.v_pi_logits_li = None

        self.g_emb = out_other['g_emb'] if out_other is not None else None
        self.bilin_emb = out_other['bilin_emb'] if out_other is not None else None
        ############################

        # self.ns = 0

        assert len(v_li) == len(set(v_li)), print(f'duplicates in v_li: {v_li}')

        # IMPORTANT: ASCENDING -> pop()
        self.v_li_valid = set(v_li)
        self.sorted_v_li_indices = self.get_sorted_v_li_vscore_li()
        if len(self.sorted_v_li_indices) > 0:
            assert self.sorted_v_li_indices[0][1] <= self.sorted_v_li_indices[-1][1], print(self.sorted_v_li_indices)
        self.solution_set = set()
        self.is_exhausted = False

        self.v_score_for_glsearch = None

    def get_sorted_v_li_vscore_li(self):
        return sorted(zip(self.v_li, self.out_policy, list(range(len(self.v_li)))), key=lambda x: x[1])

    def get_num_solutions(self):
        # print('updated:', self.ns, len(self.solution_set))
        return len(self.solution_set)

    def get_solution_coverage(self):
        return len(set().union(*[sol for sol in self.solution_set]))

    def register_nid(self, nid):
        self.nid = nid

    def get_valid_uv_pair_li(self, v_pi_li):
        v_li_valid = []
        v_pi_li_valid = []
        v_indices = []
        assert len(self.v_li) == len(v_pi_li)
        for v_idx, (v, pi) in enumerate(zip(self.v_li, v_pi_li)):
            if v in self.v_li_valid:
                v_li_valid.append(v)
                v_pi_li_valid.append(pi+1e-8)
                v_indices.append(v_idx)
        return v_li_valid, v_indices, v_pi_li_valid

    def get_next_uv_pair(self):
        # print(f'u: {self.u}, v_li: {[e[0] for e in self.sorted_v_li_indices]}')
        v, v_score, P_idx = self.sorted_v_li_indices.pop()
        self.v_li_valid.remove(v)
        assert len(self.v_li_valid) == len(self.sorted_v_li_indices)
        if len(self.sorted_v_li_indices) == 0:
            self.is_exhausted = True
        return self.u, v, v_score, P_idx

    def remove_uv_pair(self, u, v):
        assert u == self.u
        self.sorted_v_li_indices = \
            [entry for entry in self.sorted_v_li_indices if entry[0] != v]
        self.v_li_valid.remove(v)
        assert len(self.v_li_valid) == len(self.sorted_v_li_indices)
        if len(self.sorted_v_li_indices) == 0:
            self.is_exhausted = True

    def is_leaf(self):
        return self.is_dead_end or self.is_exhausted

    def number_of_states_remaining_to_explore(self):
        return len(self.v_li_valid)

    def search_depth(self):
        return 1 - len(self.nn_map) / self.gq.nx_graph.number_of_nodes()
