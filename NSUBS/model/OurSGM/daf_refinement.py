import networkx as nx

from collections import defaultdict, deque, OrderedDict
from pprint import pprint


def get_CS(G1, G2):
    assert 0 < G1.number_of_nodes() <= G2.number_of_nodes(), \
        'fix this if later we allow G1 to be data graph'
    # old2new = {G1.nodes[nid]['nid_gexf']:nid for nid in G1.nodes()}
    # nn_map = {}
    # for nid in G2.nodes():
    #     if G2.nodes[nid]['nid_gexf'] in old2new:
    #         nn_map[old2new[G2.nodes[nid]['nid_gexf']]] = nid
    # assert nx.is_isomorphic(G1.subgraph(nn_map.keys()), G2.subgraph(nn_map.values()))
    init_CS = _get_init_CS(G1, G2)
    node_ordering = _build_DAG(G1, init_CS)
    # print('num pairs (unpruned):',G1.number_of_nodes()*G2.number_of_nodes())
    CS = _refine(G1, G2, init_CS, node_ordering)
    # for u,v in nn_map.items():
    #     if v in CS[u]:
    #         print(f'ground truth pair found in CS: {u}:{v}')
    #     else:
    #         print(f'ground truth pair NOT found in CS: {u}:{v}')
    #         assert False
    # print(f'nn_map: {nn_map}')
    # print(f'all ground truth pairs found in CS!')
    # exit(-1)
    # print('num pairs in CS:',sum([len(x) for x in CS.values()]))
    # exit(-1)
    return CS, node_ordering


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


def _refine(G1, G2, init_CS, node_ordering):
    # 1st iter.
    reverse_node_ordering = _get_reverse_node_ordering(node_ordering)
    use_reverse = False
    CS = init_CS
    for _ in range(FLAGS.num_cs_refinements):
        q_D = reverse_node_ordering if use_reverse else node_ordering
        CS = _refine_iter(G1, G2, CS, q_D)
        use_reverse = not use_reverse
    return CS


def _refine_iter(G1, G2, CS, node_ordering):
    # Dynamic Programming.
    order_to_node = {v: k for k, v in node_ordering.items()}
    for order in range(G1.number_of_nodes()):
        node1 = order_to_node[order]
        node_cand_li = CS[node1].copy()  # tricky!
        # _check_prune_CS() might prune something from CS[node1], so save it
        for node2 in node_cand_li:
            CS = _check_prune_CS(node1, node2, G1, G2, CS, node_ordering)

    return CS


def _check_prune_CS(node1, node2, G1, G2, CS, node_ordering):
    # Check all node1's children. For each child, there must be at least one
    # macthable node in G2 that is conneccted to node2.
    for node1_child in _get_node1_children(node1, node_ordering, G1):
        matchable_nodes_in_G2 = CS[node1_child]
        if not _has_at_least_one_edge_from_node2_to_matchable_node(
                node2, G2, matchable_nodes_in_G2):
            CS[node1].remove(node2)
            return CS
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


################################################### path order heuristic

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


if __name__ == '__main__':
    # G1 = nx.Graph()
    # G1.add_node('u1', label='A')
    # G1.add_node('u2', label='B')
    # G1.add_node('u3', label='C')
    # G1.add_node('u4', label='D')
    # G1.add_edge('u1', 'u2')
    # G1.add_edge('u1', 'u3')
    # G1.add_edge('u2', 'u3')
    # G1.add_edge('u2', 'u4')
    # G1.add_edge('u3', 'u4')
    #
    # G2 = nx.Graph()
    # G2.add_node('v1', label='A')
    # G2.add_node('v2', label='A')
    # G2.add_node('v3', label='B')
    # G2.add_node('v4', label='B')
    # G2.add_node('v5', label='C')
    # G2.add_node('v6', label='C')
    # G2.add_node('v7', label='C')
    # G2.add_node('v8', label='B')
    # G2.add_node('v9', label='C')
    # G2.add_node('v10', label='D')
    # G2.add_node('v11', label='D')
    # G2.add_node('v12', label='B')
    # G2.add_edge('v1', 'v3')
    # G2.add_edge('v1', 'v4')
    # G2.add_edge('v1', 'v5')
    # G2.add_edge('v1', 'v6')
    # G2.add_edge('v1', 'v7')
    # G2.add_edge('v2', 'v8')
    # G2.add_edge('v2', 'v9')
    # G2.add_edge('v3', 'v5')
    # G2.add_edge('v3', 'v10')
    # G2.add_edge('v4', 'v5')
    # G2.add_edge('v4', 'v10')
    # G2.add_edge('v5', 'v10')
    # G2.add_edge('v6', 'v8')
    # G2.add_edge('v6', 'v10')
    # G2.add_edge('v7', 'v8')
    # G2.add_edge('v7', 'v10')
    # G2.add_edge('v8', 'v11')
    # G2.add_edge('v9', 'v11')
    # G2.add_edge('v9', 'v12')

    G1 = nx.Graph()
    G1.add_node('u1', label='A')
    G1.add_node('u2', label='B')
    G1.add_node('u3', label='C')
    G1.add_node('u4', label='B')
    G1.add_node('u5', label='A')
    G1.add_node('u6', label='D')
    G1.add_node('u7', label='E')
    G1.add_node('u8', label='E')
    G1.add_node('u9', label='F')
    G1.add_edge('u1', 'u2')
    G1.add_edge('u1', 'u3')
    G1.add_edge('u1', 'u4')
    G1.add_edge('u2', 'u5')
    G1.add_edge('u2', 'u6')
    G1.add_edge('u3', 'u7')
    G1.add_edge('u4', 'u8')
    G1.add_edge('u6', 'u7')
    G1.add_edge('u7', 'u9')

    G2 = nx.Graph()
    G2.add_node('v1', label='A')
    G2.add_node('v2', label='B')
    G2.add_node('v3', label='B')
    G2.add_node('v4', label='C')
    G2.add_node('v5', label='C')
    G2.add_node('v6', label='C')
    G2.add_node('v7', label='A')
    G2.add_node('v8', label='D')
    G2.add_node('v9', label='D')
    G2.add_node('v10', label='E')
    G2.add_node('v11', label='E')
    G2.add_node('v12', label='E')
    G2.add_node('v13', label='E')
    G2.add_node('v14', label='E')
    G2.add_node('v15', label='E')
    G2.add_node('v16', label='F')
    G2.add_node('v17', label='F')
    G2.add_node('v18', label='F')
    G2.add_node('v19', label='F')
    G2.add_node('v20', label='F')
    G2.add_edge('v1', 'v2')
    G2.add_edge('v1', 'v3')
    G2.add_edge('v1', 'v4')
    G2.add_edge('v1', 'v5')
    G2.add_edge('v1', 'v6')
    G2.add_edge('v2', 'v7')
    G2.add_edge('v2', 'v8')
    G2.add_edge('v2', 'v12')
    G2.add_edge('v2', 'v13')
    G2.add_edge('v2', 'v14')
    G2.add_edge('v2', 'v15')
    G2.add_edge('v3', 'v7')
    G2.add_edge('v3', 'v9')
    G2.add_edge('v4', 'v10')
    G2.add_edge('v5', 'v11')
    G2.add_edge('v6', 'v12')
    G2.add_edge('v8', 'v10')
    G2.add_edge('v8', 'v11')
    G2.add_edge('v9', 'v12')
    G2.add_edge('v10', 'v16')
    G2.add_edge('v10', 'v17')
    G2.add_edge('v11', 'v16')
    G2.add_edge('v11', 'v17')
    G2.add_edge('v11', 'v18')
    G2.add_edge('v11', 'v19')
    G2.add_edge('v11', 'v20')
    G2.add_edge('v12', 'v19')

    CS, node_ordering = get_CS(G1, G2)

    pprint(CS)
    pprint(node_ordering)
    pprint(get_daf_path_weights(G1, G2, CS, node_ordering))
