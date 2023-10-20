from copy import deepcopy
import networkx as nx
import random
import time

CUTOFF = 7200

t0 = time.time()
best_incumbent = {}
num_selections = 0

def backtrack(gq, gt, CS, daf_path_weights, nn_map):
    global num_selections
    num_selections += 1
    if num_selections % 1000 == 0:
        print(f'num_selections: {num_selections}')
    # print('start!')
    # global best_incumbent
    # if len(best_incumbent) < len(nn_map):
    #     best_incumbent = deepcopy(nn_map)

    # elif time.time() - t0 > CUTOFF:
    #     print('runtime cutoff!')
    #     return nn_map
    if len(nn_map) == gq.number_of_nodes():
        print('found solution!')
        return nn_map
    else:
        u, v_li = get_candidates(gq, gt, CS, daf_path_weights, nn_map)
        for v in v_li:
            print(f'selected action: {u} {v}, {len(v_li)} {nn_map}')
            nn_map[u] = v

            # assert nx.is_isomorphic(gq.subgraph(nn_map.keys()), gt.subgraph(nn_map.values())), \
            #     print( 'u,v', u, v,
            #            '\n u parents', CS[1].nid2node[u].get_parent_nids(),
            #            '\n v parents', [nn_map[x] for x in (CS[1].nid2node[u].get_parent_nids())],
            #            '\n u neighors', list(gq.neighbors(u)),
            #            '\n v neighors', list(gt.neighbors(v)),
            #            '\n nn_map', nn_map,
            #            '\n sanity check u (should be [])',
            #            set(nn_map.keys()).intersection(set(gq.neighbors(u))-set(CS[1].nid2node[u].get_parent_nids())),
            #            '\n sanity check v (should be [])',
            #            set(nn_map.keys()).intersection(set(gt.neighbors(v))-{nn_map[x] for x in (CS[1].nid2node[u].get_parent_nids())})
            #     )

            nn_map = backtrack(gq, gt, CS, daf_path_weights, nn_map)
            # elif time.time() - t0 > CUTOFF:
            #     print('runtime cutoff!')
            #     return nn_map
            if len(nn_map) == gq.number_of_nodes():
                print('found solution!')
                return nn_map
            else:
                del nn_map[u]
        print('backtracking!')
        assert (u is None and len(v_li) == 0) or len(v_li) > 0
        return nn_map


def get_candidates(gq, gt, CS, daf_path_weights, nn_map):
    CS_u2v, query_tree = CS
    u_next_li = \
        list(
            {u for u in gq.nodes() if
             set(query_tree.nid2node[u].get_parent_nids()).issubset(set(nn_map.keys()))} \
            - set(nn_map.keys())
        )

    score_u_vli_li = []
    # N_CANDIDATES = 0
    # for u in u_next_li:
    for u in u_next_li:
        u_parents = query_tree.nid2node[u].get_parent_nids()
        if len(u_parents) == 0:
            v_candidates = CS_u2v[u]#[v for v in CS_u2v[u] if v not in nn_map.values()]
            assert len(u_next_li) == 1
            score_u_vli_li.append((1, u, v_candidates))
        else:
            v_parents = [nn_map[u_parent] for u_parent in u_parents]
            # find all nodes in v that have v_parents
            v_candidates = list(set(CS_u2v[u]).intersection(*[set(gt.neighbors(v_parent)) for v_parent in v_parents]))
            # redundant: nodes must connect only to v_parents in nn_map and not other nodes in nn_map thats not v_parents
            v_candidates = [v for v in v_candidates if set.isdisjoint(set(gt.neighbors(v)), set(nn_map.values()) - set(v_parents))]
            v_candidates = [v for v in v_candidates if v not in nn_map.values()]
            # v_candidates = sorted(v_candidates)
            if len(v_candidates) > 0:
                score = sum([daf_path_weights[u][v] for v in v_candidates])
                score_u_vli_li.append((score, u, v_candidates))
            # N_CANDIDATES += len(v_candidates)

    if len(score_u_vli_li) == 0:
        return None, []
    else:
        _, u, v_li = min(score_u_vli_li, key=lambda x: x[0])
        return u, v_li


from data_structures_query_tree import get_levels2u2v, QueryTree
from daf_refinement import get_CS, get_daf_path_weights

gq, gt = \
    nx.read_gexf('/home/anonymous/Documents/GraphMatching/file/MCSRL_files/TODO/roadNet-CA_rw_3914_1_rw_100_1.gexf'), \
    nx.read_gexf('/home/anonymous/Documents/GraphMatching/file/MCSRL_files/TODO/roadNet-CA_rw_3914_1.gexf')
for nid in gq.nodes():
    del gq.nodes[nid]['label']
for nid in gt.nodes():
    del gt.nodes[nid]['label']
# relabel_gq = {x: i for i,x in enumerate(sorted(gq.nodes()))}
# relabel_gt = {x: i for i,x in enumerate(sorted(gt.nodes()))}
# gq = nx.relabel_nodes(gq, relabel_gq)
# gt = nx.relabel_nodes(gt, relabel_gt)
CS_u2v, node_ordering = get_CS(gq, gt)
t_CS = time.time() - t0
daf_path_weights = get_daf_path_weights(gq, gt, CS_u2v, node_ordering)
t_path_weights = time.time() - t_CS - t0
levels2u2v = get_levels2u2v((CS_u2v, node_ordering))
query_tree = QueryTree(gq,levels2u2v)
t_search_prep = time.time() - t_path_weights - t_CS - t0
nn_map = {}
backtrack(gq, gt, (CS_u2v, query_tree), daf_path_weights, nn_map)
print('----------------')
print('nn_map:', len(nn_map),',',nn_map)
# print('incumbent:', len(best_incumbent),',', best_incumbent)
is_iso = nx.is_isomorphic(gq.subgraph(nn_map.keys()), gt.subgraph(nn_map.values()))
print('num_selections:',num_selections)
print('runtime',time.time() - t0)
print('  t_CS',t_CS)
print('  t_path_weights',t_path_weights)
print('  t_search_prep',t_search_prep)
print('is_iso',is_iso)


def relabel_graphs(gq, gt):
    gq_new, gt_new = list(gq.nodes()), list(gt.nodes())
    random.shuffle(gq_new)
    random.shuffle(gt_new)
    gq = nx.relabel_nodes(gq, {old:new for (old,new) in zip(gq.nodes(), gq_new)})
    gt = nx.relabel_nodes(gt, {old:new for (old,new) in zip(gt.nodes(), gt_new)})
    return gq, gt