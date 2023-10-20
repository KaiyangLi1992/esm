from config import FLAGS
from saver import saver

from data_structures_query_tree import get_levels2u2v, QueryTree
from daf_refinement import get_CS, get_daf_path_weights
from utils_our import print_g
from utils import get_model_path, print_stats, OurTimer, format_seconds


from read_tve_imsm import read_tve
from copy import deepcopy
import networkx as nx
import random
import time
import numpy as np

from glob import glob
from pathlib import Path

from tqdm import tqdm



CUTOFF = 7200
CUT_OFF_CHEAT = -1#50

t0 = time.time()
best_incumbent = {}
nn_map_true = {}
num_selections = 0
timer = OurTimer()


def backtrack(gq, gt, CS, daf_path_weights, nn_map):
    seconds = timer.get_duration()
    if seconds > FLAGS.timeout: # more than 10 minutes!
        saver.log_info(f'Time up! {seconds} > {FLAGS.timeout}; {format_seconds(seconds)}')
        raise TimeoutError()

    global num_selections
    num_selections += 1
    if num_selections % FLAGS.print_every == 0:
        saver.log_info(f'num_selections: {num_selections}')

    # print('start!')
    global best_incumbent
    if len(best_incumbent) < len(nn_map):
        best_incumbent = deepcopy(nn_map)

    # elif time.time() - t0 > CUTOFF:
    #     print('runtime cutoff!')
    #     return nn_map
    if len(nn_map) == gq.number_of_nodes():
        # print('found solution!')
        return nn_map
    else:
        u, v_li = get_candidates(gq, gt, CS, daf_path_weights, nn_map)
        if len(nn_map) < CUT_OFF_CHEAT:
            v_true = nn_map_true[u]
            v_li.insert(0, v_li.pop(v_li.index(v_true)))
        # if num_selections == 2500:
        #     assert False
        # if '1592667' in v_li and len(nn_map) == 0:
        #     v_li = '1592667' + v_li
        for v in sorted(v_li):
            # print(f'selected action: {u} {v}, {v == nn_map_true[u]} {len(best_incumbent)} ({num_selections}), {len(v_li)} {nn_map}')
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

            # elif time.time() - t0 > CUTOFF:
            #     print('runtime cutoff!')
            #     return nn_map
            nn_map = backtrack(gq, gt, CS, daf_path_weights, nn_map)
            if len(nn_map) == gq.number_of_nodes():
                saver.log_info('found solution!')
                return nn_map
            else:
                del nn_map[u]
        # print('backtracking!')
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
    for u in sorted(u_next_li):
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
            # v_candidates = [v for v in v_candidates if set.isdisjoint(set(gt.neighbors(v)), set(nn_map.values()) - set(v_parents))]
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


def relabel_graphs(gq, gt):
    gq_new, gt_new = list(gq.nodes()), list(gt.nodes())
    random.shuffle(gq_new)
    random.shuffle(gt_new)
    gq = nx.relabel_nodes(gq, {old:new for (old,new) in zip(gq.nodes(), gq_new)})
    gt = nx.relabel_nodes(gt, {old:new for (old,new) in zip(gt.nodes(), gt_new)})
    return gq, gt


def run_single_query(gq, gt):
    nid2nid_gexf, nid_gexf2nid = {}, {}
    for nid in gq.nodes():
        nid2nid_gexf[nid] = gq.nodes[nid]['label']
        del gq.nodes[nid]['label']
    for nid in gt.nodes():
        nid_gexf2nid[gt.nodes[nid]['label']] = nid
        del gt.nodes[nid]['label'] # TODO: make it less destructive so can reuse the large target graph without relaoding
    global nn_map_true
    nn_map_true = {u: nid_gexf2nid[v] for u, v in nid2nid_gexf.items()}
    # relabel_gq = {x: i for i,x in enumerate(sorted(gq.nodes()))}
    # relabel_gt = {x: i for i,x in enumerate(sorted(gt.nodes()))}
    # gq = nx.relabel_nodes(gq, relabel_gq)
    # gt = nx.relabel_nodes(gt, relabel_gt)
    # gq, gt = relabel_graphs(gq, gt)
    CS_u2v, node_ordering = get_CS(gq, gt)
    t_CS = time.time() - t0
    daf_path_weights = get_daf_path_weights(gq, gt, CS_u2v, node_ordering)
    t_path_weights = time.time() - t_CS - t0
    levels2u2v = get_levels2u2v((CS_u2v, node_ordering))
    query_tree = QueryTree(gq, levels2u2v)
    t_search_prep = time.time() - t_path_weights - t_CS - t0
    nn_map = {}

    deg_one_nodes = [node for node, degree in dict(gq.degree()).items() if degree == 2]
    gq.remove_nodes_from(deg_one_nodes)


    # Start timer.
    try:
        timer.start_timing()
        backtrack(gq, gt, (CS_u2v, query_tree), daf_path_weights, nn_map)
    except TimeoutError:
        print('Time out; backtrack stopped')
    return nn_map, t_CS, t_path_weights, t_search_prep


def main():
    num_selections_li = []
    if FLAGS.dataset == 'road':
        # gq, gt = \
        #     nx.read_gexf('/home/anonymous/Documents/GraphMatching/file/MCSRL_files/TODO/roadNet-CA_rw_3914_1_rw_50_24.gexf'), \
        #     nx.read_gexf('/home/anonymous/Documents/GraphMatching/file/MCSRL_files/TODO/roadNet-CA_rw_3914_1_rw_50_24.gexf')
        for i in range(1,101):
            gq, gt = \
                nx.read_gexf(f'/home/anonymous/Documents/GraphMatching/file/MCSRL_files/TODO/roadNet-CA_rw_3914_1_rw_100_{i}.gexf'), \
                nx.read_gexf(f'/home/anonymous/Documents/GraphMatching/file/MCSRL_files/TODO/roadNet-CA_rw_3914_1_rw_100_{i}.gexf')

            nn_map, t_CS, t_path_weights, t_search_prep = run_single_query(gq, gt)


            saver.log_info('----------------')
            saver.log_info(f'nn_map: {len(nn_map)}, {nn_map}')
            # print('incumbent:', len(best_incumbent),',', best_incumbent)
            # is_iso = nx.is_isomorphic(gq.subgraph(nn_map.keys()), gt.subgraph(nn_map.values()))
            saver.log_info(f'num_selections: {num_selections}')
            saver.log_info(f'runtime {time.time() - t0}')
            saver.log_info(f'  t_CS {t_CS}')
            saver.log_info(f'  t_path_weights {t_path_weights}')
            saver.log_info(f'  t_search_prep {t_search_prep}')
            # print('is_iso',is_iso)
            num_selections_li.append(num_selections)
            saver.log_info(f'avg num_selections: {np.mean(np.array(num_selections_li))}')
            print_stats(num_selections_li, 'num_selections_li', print_func=saver.log_info)
    elif 'imsm-' in FLAGS.dataset:
        qgraphs_paths, ggraph_path = _get_imsm_paths()
        # gt = read_tve(ggraph_path)
        for i, qgraph in enumerate(tqdm(qgraphs_paths)):
            # i = 1
            gq = read_tve(qgraph)
            print_g(gq, f'{i}: {qgraph}', print_func=saver.log_info)

            gt = read_tve(ggraph_path)
            print_g(gt, ggraph_path,  print_func=saver.log_info)
            nn_map, t_CS, t_path_weights, t_search_prep = run_single_query(gq, gt)

            saver.log_info('----------------')
            saver.log_info(f'nn_map: {len(nn_map)}, {nn_map}')
            # print('incumbent:', len(best_incumbent),',', best_incumbent)
            # is_iso = nx.is_isomorphic(gq.subgraph(nn_map.keys()), gt.subgraph(nn_map.values()))
            saver.log_info(f'num_selections: {num_selections}')
            saver.log_info(f'runtime {time.time() - t0}')
            saver.log_info(f'  t_CS {t_CS}')
            saver.log_info(f'  t_path_weights {t_path_weights}')
            saver.log_info(f'  t_search_prep {t_search_prep}')
            # print('is_iso',is_iso)
            num_selections_li.append(num_selections)
            saver.log_info(f'avg num_selections: {np.mean(np.array(num_selections_li))}')
            print_stats(num_selections_li, 'num_selections_li', print_func=saver.log_info)
    else:
        raise NotImplementedError()

def _get_imsm_paths():
    imsm_path = f'{get_model_path()}/SubgraphMatching/dataset'
    assert 'imsm' in FLAGS.dataset
    name = FLAGS.dataset.split('imsm-')[1]
    qp = Path(f'{imsm_path}/{name}/query_graph/*.graph')
    qgraphs_paths = glob(str(qp))
    gp = Path(f'{imsm_path}/{name}/data_graph/*.graph')
    ggraph_path = glob(str(gp))
    assert len(ggraph_path) == 1
    return qgraphs_paths, ggraph_path[0]

if __name__ == '__main__':
    main()
    saver.log_info(f'Total time: {timer.time_and_clear()}')
    saver.close()