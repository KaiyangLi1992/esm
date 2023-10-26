from NSUBS.src.utils import get_model_path

import time
import networkx as nx
from collections import defaultdict, deque, OrderedDict
from pprint import pprint
from NSUBS.model.OurSGM.config import FLAGS
from tqdm import tqdm
from subprocess import Popen, PIPE
from NSUBS.model.OurSGM.saver import saver

def preproc_graph_pair(gq, gt):
    from utils import OurTimer
    timer = OurTimer()
    CS_u2v, node_ordering = get_CS(gq.path, gt.path, gq.number_of_nodes(), gt.number_of_nodes(), timer=timer)
    if timer is not None:
        timer.time_and_clear('CS', to_print=False)
    # daf_path_weights = get_daf_path_weights(gq, gt, CS_u2v, node_ordering)
    daf_path_weights = None
    if timer is not None:
        timer.time_and_clear('path weights', to_print=False)
    levels2u2v = get_levels2u2v((CS_u2v, node_ordering))
    query_tree = QueryTree(gq,levels2u2v)
    CS = CS_u2v, query_tree
    if timer is not None:
        timer.time_and_clear('levels2u2v and query tree', to_print=False)
        timer.print_durations_log(func=saver.log_info)
    return CS, daf_path_weights

def get_CS(qpn, tpn, gq_number_of_nodes=None, gt_number_of_nodes=None, timer=None):
    assert gt_number_of_nodes is None or 0 < gq_number_of_nodes <= gt_number_of_nodes, \
        'fix this if later we allow gq to be data graph'
    # init_CS = _get_init_CS(gq, gt)
    # if timer is not None:
    #     # for x, li in init_CS.items():
    #     #     print(f'{x}: {len(li)} candidates')
    #     timer.time_and_clear('init CS', to_print=False)
    # node_ordering = _build_DAG(gq, init_CS)
    # if timer is not None:
    #     timer.time_and_clear('DAG', to_print=False)


    CS, node_ordering = _refine_cpp(qpn, tpn, gq_number_of_nodes=gq_number_of_nodes)
    # hard_node_ordering = '0 1 2 3 4 5 6 7 8 9 10 11 12 13 16 17 18 19 14 23 24 21 27 26 31 20 22 15 25 29 28 30'.split(' ')
    # hard_node_ordering = {int(u):i for i,u in enumerate(hard_node_ordering)}
    # CS = _refine_python(gq, gt, init_CS, node_ordering, timer=timer)


    # saver.save_as_pickle(CS, 'CS_cpp')
    # saver.save_as_pickle(CS_python, 'CS_py')
    # assert CS_python == CS, 'different CS_python CS'
    # exit(-1)
    #
    # CS_cpp = {int(u): set([int(v) for v in v_li]) for u, v_li in CS_cpp.items()}
    # CS_python = {int(u): set([int(v) for v in v_li]) for u, v_li in CS_python.items()}

    if timer is not None:
        timer.time_and_clear('refine', to_print=False)
    return CS, node_ordering# hard_node_ordering#node_ordering

def _get_init_CS(G1, G2):
    rtn = defaultdict(list)
    for node1, ndata1 in G1.nodes(data=True):
        for node2, ndata2 in G2.nodes(data=True):
            if ndata1.get('label', 0) == ndata2.get('label', 0) and \
                    G1.degree[node1] <= G2.degree[node2]:
                rtn[node1].append(node2)
    return rtn

def _build_DAG(G, init_CS):
    source = _build_DAG_get_root(G, init_CS)

    visited = {source}
    depth_limit = len(G)
    queue = deque([(depth_limit,
                    iter(_build_DAG_sort_neighbors(G, [source])))])
    current_order = 0
    node_ordering = {source: current_order}
    parents_next_level = []
    while queue:
        depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                current_order += 1
                node_ordering[child] = current_order
                visited.add(child)
                if depth_now > 1:
                    parents_next_level.append(child)
        except StopIteration:
            if depth_now > 1:
                queue.append(
                    (depth_now - 1,
                     iter( _build_DAG_sort_neighbors(G, parents_next_level))))
                parents_next_level = []
            queue.popleft()
    return node_ordering

def _build_DAG_get_root(G, init_CS):
    li_for_sorting = []
    for node in G.nodes():
        heuristic = len(init_CS[node]) / G.degree[node]
        li_for_sorting.append((heuristic, node))
    root_node = sorted(li_for_sorting)[0][1]
    return root_node

def _build_DAG_sort_neighbors(G, cur_nodes):
    labels_frequency = defaultdict(int)
    node_labels = nx.get_node_attributes(G, 'label')
    for cur_node in cur_nodes:
        for node in G.neighbors(cur_node):
            label = node_labels.get(node, 0)
            labels_frequency[label] += 1

    neighbors = []
    for cur_node in cur_nodes:
        for node in G.neighbors(cur_node):
            label = node_labels.get(node, 0)
            neighbors.append((labels_frequency[label], G.degree[node], node))

    neighbors.sort()
    neighbors = [n[2] for n in neighbors]
    return neighbors


def _refine_cpp(qpn, tpn, gq_number_of_nodes=None):
    addi = ''
    if 'serverxxx' in FLAGS.hostname or 'serverxxx' in FLAGS.hostname:
        addi = f'_{FLAGS.hostname}'

    execution_args = generate_args(
        f'{get_model_path()}/SubgraphMatching/build{addi}/matching/SubgraphMatching.out',
        '-d', tpn, '-q', qpn, '-filter', 'DPiso',
        '-order', 'DPiso', '-engine', 'DPiso', '-num', '1')

    print(f'{get_model_path()}/SubgraphMatching/build{addi}/matching/SubgraphMatching.out')
    print('------------------------')

    print('@@@@ ', ' '.join(execution_args))

    (rc, std_output, std_error) = execute_binary(execution_args)
    std_output = std_output.decode("utf-8")
    # print('------------------------')
    # print(str(std_output))
    #
    # print('------------------------')
    # print(str(std_output).split('&&&')[-2])
    # print('------------------------')
    print(f'std_out_len: {len(str(std_output))}')
    # _, CS, _, node_ordering, _ = eval(str(std_output).split('&&&')[-2])
    print(len(str(std_output).split('&&&')))
    # print(str(std_output))
    assert '&&&' in str(std_output), f'C++ code output:\n {std_output}'
    _, CS, _, node_ordering, _ = str(std_output).split('&&&')
    CS = eval(CS)
    node_ordering = node_ordering.split(' ')[2:-1]
    # print(node_ordering)
    node_ordering = {int(u):i for i,u in enumerate(node_ordering)}
    assert gq_number_of_nodes is None or len(node_ordering) == gq_number_of_nodes
    return CS, node_ordering


def generate_args(binary, *params):
    arguments = [binary]
    arguments.extend(list(params))
    return arguments
'''
logs/our_train_imsm-patents_list_nn_300_2022-05-11T00-09-37.672436/config.py
'''

def execute_binary(args):
    process = Popen(' '.join(args), shell=True, stdout=PIPE, stderr=PIPE)
    (std_output, std_error) = process.communicate()
    process.wait()
    rc = process.returncode

    return rc, std_output, std_error



def _refine_python(G1, G2, init_CS, node_ordering, timer=None):
    num_cs_refinements = FLAGS.num_cs_refinements
    assert 0 <= num_cs_refinements <= 3
    if num_cs_refinements == 0:
        return init_CS
    # 1st iter.
    reverse_node_ordering = _get_reverse_node_ordering(node_ordering)
    if timer is not None:
        timer.time_and_clear('reverse_node_ordering', to_print=False)
    CS = _refine_iter(G1, G2, init_CS, node_ordering, timer)
    if timer is not None:
        timer.time_and_clear('1st iter', to_print=False)
    if num_cs_refinements == 1:
        return CS
    # 2nd iter.
    CS = _refine_iter(G1, G2, CS, reverse_node_ordering)
    if timer is not None:
        timer.time_and_clear('2nd iter', to_print=False)
    if num_cs_refinements == 2:
        return CS
    # 3rd iter.
    CS = _refine_iter(G1, G2, CS, node_ordering)
    if timer is not None:
        timer.time_and_clear('3rd iter', to_print=False)
    if num_cs_refinements == 1:
        return CS
    return CS

def _refine_iter(G1, G2, CS, node_ordering, timer=None):
    # Dynamic Programming.
    order_to_node = {v: k for k, v in node_ordering.items()}
    for order in tqdm(range(G1.number_of_nodes())):
        if timer is not None:
            timer.time_and_clear(f'order {order}', to_print=False)

        node1 = order_to_node[order]
        node_cand_li = CS[node1].copy()  # tricky!

        if timer is not None:
            timer.time_and_clear(f'order {order} copied {len(node_cand_li)}', to_print=False)
        # _check_prune_CS() might prune something from CS[node1], so save it
        for node2 in node_cand_li:
            CS = _check_prune_CS(node1, node2, G1, G2, CS, node_ordering, timer)
        if timer is not None:
            timer.time_and_clear(f'order {order} for done', to_print=False)

    return CS

def _check_prune_CS(node1, node2, G1, G2, CS, node_ordering, timer=None):
    # Check all node1's children. For each child, there must be at least one
    # macthable node in G2 that is conneccted to node2.
    tmp = _get_node1_children(node1, node_ordering, G1)
    # if timer is not None:
    #     timer.time_and_clear(f'node1 {node1} node2 {node2} _get_node1_children', to_print=False)

    for node1_child in tmp:
        matchable_nodes_in_G2 = CS[node1_child]
        if not _has_at_least_one_edge_from_node2_to_matchable_node(
                node2, G2, matchable_nodes_in_G2):
            CS[node1].remove(node2)
            # if timer is not None:
                # timer.time_and_clear(f'node1 {node1} node2 {node2} loop over {len(tmp)} but return early', to_print=False)
            return CS

    # if timer is not None:
    #     timer.time_and_clear(f'node1 {node1} node2 {node2} loop over {len(tmp)}', to_print=False)
    # No pruning :)
    return CS

def _get_node1_children(node1, node_ordering, G1):
    # If node1_order == 0, return [].
    # Otherwise, return all the children of node1 that are connected to node1.
    node1_order = node_ordering[node1]
    if node1_order == 0:
        return []
    rtn = []
    for neighbor in G1.neighbors(node1):
        if node_ordering[neighbor] < node1_order:
            rtn.append(neighbor)
    return rtn

def _has_at_least_one_edge_from_node2_to_matchable_node(
        node2, G2, matchable_nodes_in_G2):
    for mnG2 in matchable_nodes_in_G2:
        if G2.has_edge(node2, mnG2):
            return True
    return False

def _get_reverse_node_ordering(node_ordering):
    return {k: len(node_ordering) - 1 - v for k, v in node_ordering.items()}

def get_daf_path_weights(G1, G2, CS, node_ordering):
    reverse_node_ordering = _get_reverse_node_ordering(node_ordering)
    order_to_node = {v: k for k, v in reverse_node_ordering.items()}
    rtn = defaultdict(OrderedDict)
    # 1st run: base cases.
    for order in range(G1.number_of_nodes()):
        node1 = order_to_node[order]
        all_cs = _get_children_with_only_one_parent(node1, G1, node_ordering)
        if len(all_cs) == 0:
            for node2 in CS[node1]:
                rtn[node1][node2] = 1
    # 2nd run.
    for order in range(G1.number_of_nodes()):
        node1 = order_to_node[order]
        if node1 not in rtn:
            all_cs = _get_children_with_only_one_parent(node1, G1, node_ordering)
            assert len(all_cs) != 0
            for node2 in CS[node1]:
                child1s_weights = []
                for child1 in all_cs:
                    to_sum = []
                    for child2 in CS[child1]:
                        if G2.has_edge(node2, child2):
                            to_sum.append(rtn[child1][child2])
                    assert len(to_sum) > 0, 'Something wrong with CS'
                    child1s_weights.append(sum(to_sum))
                assert len(child1s_weights) > 0, \
                    'Something wrong in _get_children_with_only_one_parent()'
                rtn[node1][node2] = min(child1s_weights)
    return rtn

def _get_children_with_only_one_parent(node, G, node_ordering):
    children = _get_children(node, G, node_ordering)
    rtn = []
    for child in children:
        parents_of_child = _get_parents(child, G, node_ordering)
        if len(parents_of_child) == 1:
            rtn.append(child)
    return rtn

def _get_children(node, G, node_ordering):
    return _get_children_parents_helper(node, G, node_ordering, True)

def _get_parents(node, G, node_ordering):
    return _get_children_parents_helper(node, G, node_ordering, False)

def _get_children_parents_helper(node, G, node_ordering, need_child):
    node_o = node_ordering[node]
    rtn = []
    for neighbor in G.neighbors(node):
        if need_child:
            if node_ordering[neighbor] > node_o:
                rtn.append(neighbor)
        else:
            if node_ordering[neighbor] < node_o:
                rtn.append(neighbor)
    return rtn

class QueryNode():
    def __init__(self, nid, level, parents):
        self.nid = nid
        self.level = level
        self.parents = parents
        self.children = []
    def get_parent_nids(self):
        return [parent.nid for parent in self.parents]
    def get_one_level_lower_children(self):
        return [child for child in self.children if child.level == self.level+1]

class QueryTree():
    def __init__(self, g_query, levels2u2v):
        assert 0 in levels2u2v
        assert 0 == min(levels2u2v.keys())
        for level, u2v in sorted(levels2u2v.items()):
            if level == 0:
                u_root, = u2v.keys()
                self.root = QueryNode(u_root, 0, [])
                self.nid2node = {u_root: self.root}
            else:
                u_li = u2v.keys()
                for u in u_li:
                    nid_li_parents = set.intersection(set(g_query.neighbors(u)), set(self.nid2node.keys()))
                    self.nid2node[u] = \
                        QueryNode(u, level, [self.nid2node[nid] for nid in nid_li_parents])
                    for u_parent in nid_li_parents:
                        self.nid2node[u_parent].children.append(self.nid2node[u])

from collections import defaultdict
def get_levels2u2v(CS):
    u2v, u2levels = CS
    levels2u2v = defaultdict(dict)
    for u,level in u2levels.items():
        levels2u2v[level][u] = u2v[u]
    return levels2u2v
