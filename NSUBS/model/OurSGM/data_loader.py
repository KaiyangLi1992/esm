from NSUBS.model.OurSGM.utils_our import get_flags_with_prefix_as_list
from NSUBS.model.OurSGM.read_tve_imsm import read_tve
from NSUBS.model.OurSGM.write_tve_imsm import write_tve
from NSUBS.model.OurSGM.utils_our import print_g, get_save_path, get_flag
from NSUBS.src.utils import get_model_path, OurTimer, create_dir_if_not_exists, \
    save_pickle, load_pickle, sorted_nicely, print_stats
from NSUBS.model.OurSGM.sgm_graph import SRW_RWF_ISRW, random_walk_sampling, \
    convert_node_labels_to_integers_with_mapping, OurGraph, OurGraphPair
from NSUBS.model.OurSGM.daf import preproc_graph_pair

import torch
import numpy as np
import networkx as nx
import pickle
from glob import glob
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import to_networkx
from random import Random
from os.path import join, isfile, dirname
from copy import deepcopy
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.classic import wheel_graph


def _load_CS_dpw(load_path):
    if load_path is not None:
        ld = load_pickle(load_path)
        if ld is not None:
            CS, daf_path_weights = ld['CS'], ld['dpw']
            saver.log_info(f'Loaded CS and daf_path_weights from {load_path}')
            return True, (CS, daf_path_weights)
    return False, (None, None)


def _print_CS_stats(CS):
    x = [len(v) for v in CS[0].values()]
    print_stats(x, 'CS_cand_set_size', print_func=saver.log_info)


def get_CS_wrapper(gq, gt, load_path=None):
    loaded, out = _load_CS_dpw(load_path)
    if loaded:
        CS, daf_path_weights = out
    else:
        timer = OurTimer()
        timer.time_and_clear('OurGraph created')
        CS, daf_path_weights = preproc_graph_pair(gq, gt)
        timer.time_and_clear('CS, daf_path_weights')
        timer.print_durations_log()
        if load_path is not None:
            save_pickle({'CS': CS, 'dpw': daf_path_weights},
                        load_path, print_msg=False)
            saver.log_info(f'Saved pair to {load_path}')
    _print_CS_stats(CS)
    return CS, daf_path_weights


def get_data_loader_wrapper(tvt):
    # if tvt == 'train':
    data_loader = data_loader_wrapper(tvt)
    # else:
    #     data_loader = get_data_loader(*get_data_loader_params(tvt))
    return data_loader


def get_data_loader(dataset_name, subgroup_li, subgroup, path_indices, sample_size=None):
    if 'imsm-' in dataset_name:
        if subgroup == 'list':
            data_loader = get_data_loader_imsm_merged(
                dataset_name, subgroup_li, path_indices, sample_size)
        else:
            assert subgroup_li is None
            data_loader = get_data_loader_imsm(
                dataset_name, subgroup, path_indices, sample_size)
    elif dataset_name == 'custom':
        assert False, 'Are we really using custom?'
        val_loader = get_data_loader_custom(FLAGS.custom_conf)
    elif dataset_name == 'mix':
        data_loader = get_mix_loader()
    else:
        assert False, f'{dataset_name} deprecated or not implemented!'
    return data_loader


def get_data_loader_params(tvt):
    if tvt == 'train':
        dataset_name, subgroup_li, subgroup, path_indices = \
            FLAGS.train_dataset, FLAGS.train_subgroup_list, FLAGS.train_subgroup, FLAGS.train_path_indices
    elif tvt == 'val':
        dataset_name, subgroup_li, subgroup, path_indices = \
            FLAGS.val_dataset, FLAGS.val_subgroup_list, FLAGS.val_subgroup, FLAGS.val_path_indices
    elif tvt == 'test':
        dataset_name, subgroup_li, subgroup, path_indices = \
            FLAGS.test_dataset, FLAGS.test_subgroup_list, FLAGS.test_subgroup, FLAGS.test_path_indices
    else:
        assert False
    return dataset_name, subgroup_li, subgroup, path_indices


def data_loader_wrapper(tvt):
    if tvt == 'train':
        sample_size_li = FLAGS.train_sample_size_li
    elif tvt == 'val':
        sample_size_li = FLAGS.val_sample_size_li
    elif tvt == 'test':
        sample_size_li = [None]
    else:
        assert False
    for sample_size in sample_size_li:
        data_loader = get_data_loader(*get_data_loader_params(tvt), sample_size=sample_size)
        for gp in data_loader:
            yield gp


def confirm_data_leakage():
    consensus = None
    while consensus is None:
        c = input('allow data leakage (y/n)?')
        if c in ['y', 'n']:
            consensus = c
        else:
            print('type either "y" or "n"')
    if consensus == 'y':
        pass
    elif consensus == 'n':
        assert False
    else:
        assert False


def relabel_CS(CS, node_map):
    CS_u2v, query_tree = CS
    CS_u2v_filtered = {}
    for u, v_li in CS_u2v.items():
        v_li_filtered = [node_map[v] for v in v_li if v in node_map]
        if len(v_li_filtered) > 0: CS_u2v_filtered[u] = v_li_filtered
    # TODO: do we need to relabel query_tree??
    return CS_u2v_filtered, query_tree


def relabel_daf_path_weights(daf_path_weights, node_map):
    # daf_path_weights_filtered = {}
    for u, v2weight in daf_path_weights.items():
        v2weight_filtered = \
            {node_map[v]: weight for (v, weight) in v2weight.items() if v in node_map}
        if len(v2weight_filtered) > 0: daf_path_weights_filtered[u] = v2weight_filtered
    return daf_path_weights_filtered


def get_edge_index(g):
    g_edges = [[int(x) for x in lst] for lst in g.edges()]
    edge_index = torch.LongTensor(g_edges).t().contiguous().to(FLAGS.device)
    return edge_index


def filter_target_graph(g, CS, daf_path_weights, sel_nodes, true_nn_map):
    assert len(sel_nodes) == len(set(sel_nodes))
    relabel_dict = {nid: i for (i, nid) in enumerate(sel_nodes)}
    g_filtered = nx.subgraph(g, sel_nodes)
    g_filtered = nx.relabel_nodes(g_filtered, relabel_dict)
    CS_filtered = relabel_CS(CS, relabel_dict)
    daf_path_weights_filtered = None
    # daf_path_weights_filtered = relabel_daf_path_weights(daf_path_weights, relabel_dict)
    # Relabel true_nn_map.
    true_nn_map_new = {}
    if true_nn_map is not None:
        for node1, node2 in true_nn_map.items():
            true_nn_map_new[node1] = relabel_dict[node2]
    return g_filtered, CS_filtered, daf_path_weights_filtered, true_nn_map_new, relabel_dict


def get_data_loader_imsm_merged(dataset, subgroup_list, path_indices_li, sample_size):
    for subgroup, path_indices in zip(subgroup_list, path_indices_li):
        dl = get_data_loader_imsm(dataset, subgroup, path_indices, sample_size)
        for i, gp in enumerate(dl):
            saver.log_info(f'Merged loader: {subgroup} {i}')
            yield gp


def _get_gt_sampled(gt, sample_size, srw_rwf_isrw, pkl_folder, suffix_gt_sample):
    pn = join(pkl_folder, f'target_sampled{suffix_gt_sample}.pickle')
    ld = load_pickle(pn)
    if ld is not None:
        gt = ld['gt']
        saver.log_info(f'Loaded target sampled target graph from {pn}')
    else:
        gt = random_walk_sampling(gt, sample_size, srw_rwf_isrw, check_connected=False)
        gt = nx.convert_node_labels_to_integers(gt)
        save_pickle({'gt': gt}, pn, print_msg=False)
        saver.log_info(f'Saved target sampled graph (OurGraph) to {pn}')
    return gt

def get_data_loader_imsm(dataset, subgroup, path_indices, sample_size):
    qgraphs_paths, tpn, pkl_folder = _get_imsm_paths(
        dataset, subgroup, path_indices)
    create_dir_if_not_exists(pkl_folder)
    gt, _, enc = _get_g_X_enc(pkl_folder, tpn)
    enc = None if (hasattr(FLAGS, 'no_one_hot') and FLAGS.no_one_hot) else enc
    gt.path = tpn

    if 'sample' in subgroup or sample_size is not None:
        srw_rwf_isrw = SRW_RWF_ISRW(Random(345))
        gsize = int(subgroup.split('_')[1])
        # if not nx.is_connected(gt):
        # print_g(gt, 'gt ia not connected', print_func=saver.log_info)
        # gt = max(nx.connected_component_subgraphs(gt), key=len)
        # print_g(gt, 'gt ia now connected', print_func=saver.log_info)
        # assert nx.is_connected(gt)
    else:
        srw_rwf_isrw = None
        gsize = None

    if sample_size is None or sample_size > gt.number_of_nodes():
        suffix_gt_sample = ''
    else:
        suffix_gt_sample = f'_sample={sample_size}'
        gt = _get_gt_sampled(gt, sample_size, srw_rwf_isrw, pkl_folder, suffix_gt_sample)
        tpn_sample = f'{tpn}{suffix_gt_sample}'
        write_tve(gt, tpn_sample)
        gt.path = tpn_sample

    if FLAGS.num_cs_refinements == 3:
        suffix_cs_refinement = ''
    else:
        suffix_cs_refinement = f'_{FLAGS.num_cs_refinements}refined'

    for i in tqdm(range(len(qgraphs_paths))):
        true_nn_map = None
        if 'sample' in subgroup:
            sample_qpath = join(pkl_folder, f'sample_{i}_gq.pickle')
            gq, true_nn_map = _get_sampled_gq(gt, gsize, srw_rwf_isrw, sample_qpath)
            name = 'sampled'
            query_id = i
            gq.path = None
        else:
            qpn = qgraphs_paths[i]
            gq = read_tve(qpn)
            name = qpn
            query_id = int(qpn.split('.graph')[0].split('_')[-1]) - 1  # 1-based --> 0-based
            assert 0 <= query_id <= 199, f'{query_id}'
            gq.path = qpn

        fn = f'{query_id}p{suffix_cs_refinement}{suffix_gt_sample}.pickle'
        ppn = join(pkl_folder, fn)
        if hasattr(FLAGS, 'no_one_hot') and FLAGS.no_one_hot:
            suffix_no_one_hot = '_n-oh'
        else:
            suffix_no_one_hot = ''
        load_path = join(pkl_folder, f'{fn.split(".")[0]}{suffix_no_one_hot}_pqf.pickle')

        gp, gq, gt_filtered = _gen_load_CS_gp_filter(load_path, gq, gt, true_nn_map, ppn, enc)

        saver.log_info(f'\n\n--- Query {i}/{len(qgraphs_paths)}: {name}')
        print_g(gq.nx_graph, 'gq', print_func=saver.log_info)
        print_g(gt_filtered.nx_graph, 'gt', print_func=saver.log_info)

        # saver.save_graph_as_gexf(_change_label(gq), f'{basename(qpn)}.gexf')
        yield gp


def _gen_load_CS_gp_filter(load_path, gq, gt, true_nn_map, ppn, enc):
    loaded, out = _load_CS_gp_filter(load_path)
    if loaded:
        gp, gq, gt_filtered = out
    else:
        CS, daf_path_weights = get_CS_wrapper(gq, gt, load_path=ppn)
        selected_nodes = set([v for v_li in CS[0].values() for v in v_li])
        gt_filtered, CS_filtered, daf_path_weights_filtered, true_nn_map, relabel_dict = \
            filter_target_graph(
                gt, CS, daf_path_weights,
                selected_nodes, true_nn_map
            )
        gt_filtered = OurGraph(gt_filtered, encoder=enc)
        gq = OurGraph(gq, encoder=enc)
        gp = \
            OurGraphPair(
                gq, gt_filtered, CS_filtered,
                daf_path_weights_filtered,
                encoder=enc, true_nn_map=true_nn_map,
                relabel_dict=relabel_dict
            )
        if load_path is not None:
            save_pickle({'gp': gp, 'gq': gq, 'gt_filtered': gt_filtered},
                        load_path, print_msg=False)
            saver.log_info(f'Saved gp, gq, gt_filtered to {load_path}')

    return gp, gq, gt_filtered


def _load_CS_gp_filter(load_path):
    if load_path is not None:
        ld = load_pickle(load_path)
        if ld is not None:
            gp, gq, gt_filtered = ld['gp'], ld['gq'], ld['gt_filtered']
            saver.log_info(f'Loaded gp, gq, gt_filtered from {load_path}')
            return True, (gp, gq, gt_filtered)

    return False, None


def _get_sampled_gq(gt, gsize, srw_rwf_isrw, load_path):
    ld = load_pickle(load_path)
    if ld is not None:
        gq, true_nn_map = ld['gq'], ld['true_nn_map']
        saver.log_info(f'Sampled gq: Loaded gq and true_nn_map from {load_path}')
    else:
        gq = random_walk_sampling(gt, gsize, srw_rwf_isrw, check_connected=False)
        # node labels/colors are copied/sampled from the original target graph
        gq, true_nn_map = \
            convert_node_labels_to_integers_with_mapping(
                gq)  # TODO: save the node-node mappings for pretraining?
        # print(gq._adj)
        # print(true_nn_map)
        # exit(-1)
        save_pickle({'gq': gq, 'true_nn_map': true_nn_map},
                    load_path, print_msg=False)
        saver.log_info(f'Sampled gq: Saved gq and true_nn_map to {load_path}')
    return gq, true_nn_map

def _create_fit_onehotencoder(gt):
    enc = OneHotEncoder()
    X = []
    X_flat = []
    for node, ndata in sorted(gt.nodes(data=True)):
        X.append([ndata['type']])
        X_flat.append(ndata['type'])
    # print_stats(X_flat, 'node labels', print_func=saver.log_info)
    enc.fit(X)
    # saver.log_info(f'Fit one hot encoder')
    return enc, X

def _get_enc_X(gt):
    enc, X = _create_fit_onehotencoder(gt)
    return enc, X


def _get_g_X_enc(pkl_folder, tpn):
    g_X_enc_pn = join(pkl_folder, f'target.pickle')
    # ld = load_pickle(g_X_enc_pn)
    ld = None
    if ld is not None:
        gt, X, enc = ld['gto']
        # # assert type(gto) is OurGraph
        # # THere may be version mismatch across different versions of sklearn.
        # # Thus, to ensure consistency, re-create and fit the encoder using gt.
        # enc, X = _get_enc_X(gt)

        saver.log_info(f'Loaded target graph, initial features, and encoder from {g_X_enc_pn}')
    else:
        timer = OurTimer()
        gt = read_tve(tpn)
        enc, X = _get_enc_X(gt)

        timer.time_and_clear('read_tve')
        gt_X_enc = (gt, X, enc)
        timer.time_and_clear('OurGraph')
        save_pickle({'gto': gt_X_enc}, g_X_enc_pn, print_msg=False)
        saver.log_info(f'Saved target graph (OurGraph) to {g_X_enc_pn}')
    return gt, X, enc


 


def _get_imsm_paths(dataset, subgroup, path_indices):
    data_loader_path = get_flag('special_data_loader_path')
    if data_loader_path is None:
        data_loader_path = get_model_path()
    imsm_path = f'{data_loader_path}/SubgraphMatching/dataset'
    assert 'imsm' in dataset
    name = dataset.split('imsm-')[1]
    qp = Path(f'{imsm_path}/{name}/query_graph/query_{subgroup}_*.graph')
    qgraphs_paths = sorted_nicely(glob(str(qp)))
    len_qgraphs_paths = len(qgraphs_paths)

    num_pairs = max(path_indices) - min(path_indices)
    if 'sample' not in subgroup:
        assert num_pairs <= len_qgraphs_paths, \
            f'{num_pairs} {len_qgraphs_paths} {qgraphs_paths} {dataset}, {subgroup}, ' \
                f'{num_pairs} qp={qp}'
    indices_start, indices_end = path_indices
    qgraphs_paths = qgraphs_paths[indices_start:indices_end]
    if 'sample' not in subgroup:
        assert len(qgraphs_paths) == num_pairs
        saver.log_info(f'Take {num_pairs} out of {len_qgraphs_paths} query graphs')
    # print(qgraphs_paths)

    gp = Path(f'{imsm_path}/{name}/data_graph/*.graph')
    ggraph_path = glob(str(gp))
    assert len(ggraph_path) == 1
    if data_loader_path is None:
        save_path = get_save_path()
    else:
        save_path = join(data_loader_path, 'OurSGM', 'save')
    return sorted(qgraphs_paths), ggraph_path[0], join(save_path, f'{dataset}_{subgroup}')


def get_data_loader_default(pn, num_samples=999):
    assert False
    for i in range(num_samples):
        ppn, qpn, tpn = join(pn, f'{i}p.pkl'), join(pn, f'{i}q.gexf'), join(pn, f'{i}t.gexf')
        if i > num_samples or not (isfile(qpn) and isfile(tpn)):
            break
        if isfile(ppn):
            with open(ppn, 'rb') as fp:
                gp = pickle.load(fp)
            gq, gt = gp.gq.nx_graph, gp.gt.nx_graph
        else:
            gq, gt = nx.read_gexf(qpn), nx.read_gexf(tpn)
            # gp = OurGraphPair(gq, gt)
            with open(ppn, 'wb') as fp:
                pickle.dump(gp, fp)
        saver.log_info(f'--- Query {i}/{num_samples}')
        print_g(gq, 'gq', print_func=saver.log_info)
        print_g(gt, 'gt', print_func=saver.log_info)
        yield gp


def get_mix_loader(no_DAF=False):  # from pytorch geometric
    srw_rwf_isrw = SRW_RWF_ISRW(Random(345))

    query_id = 0
    for i, tup in enumerate(FLAGS.mix):
        assert type(tup) is tuple and len(tup) == 3
        dataset, num_graphs, gsizes = tup
        # print(dataset, num_graphs, gsizes)
        root = join(get_save_path(), 'mix_targets')
        saver.log_info('\n\n')
        gto = _load_mix_dataset_target_graph(dataset, root)
        gt = gto.nx_graph

        pkl_folder = join(root, dataset)

        # dataset_better_s = dataset.replace(':', '-')
        gt_path = join(pkl_folder, f'{dataset}.graph')
        gt.path = gt_path
        create_dir_if_not_exists(dirname(gt_path))
        write_tve(gt, gt_path)

        saver.log_info(f'--- Target {i}/{len(FLAGS.mix)}: {dataset}')
        print_g(gt, f'gt:{dataset}', print_func=saver.log_info)  # TODO: node labels?
        assert nx.number_connected_components(gt) == 1


        for j in range(num_graphs):
            name = f'{i}_{dataset}_{j}'
            if type(gsizes) is list:
                gsize = gsizes[j]
            else:
                assert type(gsizes) is int
                gsize = gsizes
            gq = random_walk_sampling(gt, gsize, srw_rwf_isrw)
            gq, true_nn_map = convert_node_labels_to_integers_with_mapping(
                gq)  # TODO: save the node-node mappings for pretraining?
            print_g(gq, f'gq:{dataset}-{j}', print_func=saver.log_info)

            gq_path = join(pkl_folder, f'{name}.graph')
            gq.path = gq_path
            create_dir_if_not_exists(dirname(gq_path))
            write_tve(gq, gq_path)


            if no_DAF:  # for debugging
                yield gq, gt
            else:
                # gp = OurGraphPair(gq, gto, true_nn_map=true_nn_map)  # TODO: update
                sample_size = gsize
                # if sample_size is None or sample_size > gt.number_of_nodes():
                #     suffix_gt_sample = ''
                # else:
                #     suffix_gt_sample = f'_sample={sample_size}'
                #     gt = _get_gt_sampled(gt, sample_size, srw_rwf_isrw, root,
                #                          suffix_gt_sample)
                #     tpn_sample = f'{tpn}{suffix_gt_sample}'
                #     write_tve(gt, tpn_sample)
                #     gt.path = tpn_sample


                if FLAGS.num_cs_refinements == 3:
                    suffix_cs_refinement = ''
                else:
                    suffix_cs_refinement = f'_{FLAGS.num_cs_refinements}refined'
                fn = f'{query_id}{name}p{suffix_cs_refinement}.pickle'


                ppn = join(pkl_folder, fn)
                if hasattr(FLAGS, 'no_one_hot') and FLAGS.no_one_hot:
                    suffix_no_one_hot = '_n-oh'
                else:
                    suffix_no_one_hot = ''
                load_path = join(pkl_folder, f'{fn.split(".")[0]}{suffix_no_one_hot}_pqf.pickle')

                gp, gq, gt_filtered = _gen_load_CS_gp_filter(load_path, gq, gt, true_nn_map, ppn,
                                                             enc=None)

                saver.log_info(f'\n\n--- Query {query_id}: {name}')
                print_g(gq.nx_graph, 'gq', print_func=saver.log_info)
                print_g(gt_filtered.nx_graph, 'gt', print_func=saver.log_info)
                query_id += 1

                yield gp


def _load_mix_dataset_target_graph(dataset, root):
    # assert False  # follow format of imsm
    # g_X_enc_pn = join(pkl_folder, f'target.pickle')
    # ld = load_pickle(g_X_enc_pn)
    # if ld is not None:
    #     gt, X, enc = ld['gto']
    #     # # assert type(gto) is OurGraph
    #     # # THere may be version mismatch across different versions of sklearn.
    #     # # Thus, to ensure consistency, re-create and fit the encoder using gt.
    #     # enc, X = _get_enc_X(gt)
    #
    #     saver.log_info(f'Loaded target graph, initial features, and encoder from {g_X_enc_pn}')
    # else:
    #     timer = OurTimer()
    #     gt = read_tve(tpn)
    #     enc, X = _get_enc_X(gt)
    #
    #     timer.time_and_clear('read_tve')
    #     gt_X_enc = (gt, X, enc)
    #     timer.time_and_clear('OurGraph')
    #     save_pickle({'gto': gt_X_enc}, g_X_enc_pn, print_msg=False)
    #     saver.log_info(f'Saved target graph (OurGraph) to {g_X_enc_pn}')
    create_dir_if_not_exists(root)
    ds = dataset.replace(':', '-')
    gto_pn = join(root, f'{ds}_target.pickle')
    ld = load_pickle(gto_pn)
    if ld is not None:
        gto = ld['gto']
        assert type(gto) is OurGraph
        saver.log_info(f'Loaded target graph (OurGraph) {gto_pn}')
    else:
        gto = _load_gto_pyg(dataset)
        save_pickle({'gto': gto}, gto_pn, print_msg=False)
        saver.log_info(f'Saved target graph (OurGraph) to {gto_pn}')
    return gto


def _load_gto_pyg(dataset):
    from torch_geometric.datasets import PPI, KarateClub
    # obj = PPI(get_save_path(), split='train', transform=None, pre_transform=None, pre_filter=None)
    # print(obj)

    if 'BA-' in dataset:
        dss = dataset.split('-')
        n, m = int(dss[1]), int(dss[2])
        gt = nx.barabasi_albert_graph(n=n,
                                      m=m,
                                      seed=0)
        gt = nx.convert_node_labels_to_integers(gt)
        return OurGraph(gt)
    elif 'ER-' in dataset:
        dss = dataset.split('-')
        n, p = int(dss[1]), float(dss[2])
        gt = nx.erdos_renyi_graph(n=n,
                                      p=p,
                                      seed=0)
        gt = nx.convert_node_labels_to_integers(gt)
        return OurGraph(gt)
    elif 'WS-' in dataset:
        dss = dataset.split('-')
        n, k, p = int(dss[1]), int(dss[2]), float(dss[3])
        gt = nx.watts_strogatz_graph(n=n,
                                     k=k,
                                      p=p,
                                      seed=0)
        gt = nx.convert_node_labels_to_integers(gt)
        return OurGraph(gt)

    import importlib
    m = importlib.import_module('torch_geometric.datasets')
    # get the class, will raise AttributeError if class cannot be found

    if '-' in dataset:
        ds = dataset.split('-')
        class_name = ds[0]
        if len(ds) >= 2:
            sub_name = '-'.join(ds[1:])
        else:
            sub_name = ds[1]
    else:
        class_name, sub_name = dataset, ''

    if class_name == 'BioSNAPDataset':
        from more_dataset import BioSNAPDataset
        c = BioSNAPDataset
    else:
        c = getattr(m, class_name)
    saver.log_info(f'Use {class_name}{": " + sub_name if sub_name != "" else ""} -- {c}')

    root = join(get_save_path(), dataset.replace(':', '-'))

    vnames = c.__init__.__code__.co_varnames
    if 'root' in vnames:
        if 'name' in vnames:
            if 'feature' in vnames:
                data = c(root=root, name=sub_name, feature='profile')  # feature does not matter
            elif 'tissue' in vnames:
                ss = sub_name.split(',')
                data = c(root=root, name=ss[0], tissue=ss[1])
            else:
                print('sub_name', sub_name)
                data = c(root=root, name=sub_name)
        elif 'pair' in vnames:
            data = c(root=root, pair=sub_name)
        elif 'homophily' in vnames:
            data = c(root=root, homophily=float(sub_name))
        else:
            data = c(root=root)
    else:
        assert 'name' not in vnames
        data = c()

    saver.log_info(f'Loaded pyg data: {data} {data.data}')

    G = to_networkx(data.data)  # data.data may have different x encoding
    G = nx.Graph(G)  # undirected
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    saver.log_info(f'{len(components)} components with largest cc '
                   f'{len(components[0])} (/{len(G)}={len(components[0]) / len(G):.4%})')
    G = G.subgraph(components[0])
    G = nx.convert_node_labels_to_integers(G)
    return OurGraph(G)  # encoding will be consistent


########################## Custom

def get_data_loader_custom(custom_conf):
    if custom_conf['target'] == 'grid2d':
        gt = nx.convert_node_labels_to_integers(
            grid_2d_graph(custom_conf['m'], custom_conf['n']))
        if custom_conf['query'] == 'triangle':
            gq = _gen_triangle()
            gt.add_edge(0, 2)
            true_nn_map = {0: 0, 1: 1, 2: 2}  # only 1 solution globally
            # gt.add_edge(154, 203)
            # true_nn_map = {0: 153, 1: 154, 2: 203}
            # true_nn_map = None
        elif custom_conf['query'] == 'hexagon':
            gq = _gen_hexagon()
            gt.add_edge(0, 5)
            true_nn_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # many more solutions
        elif custom_conf['query'] == 'heptagon':
            gq = _gen_heptagon()
            gt.add_edge(0, 6)
            true_nn_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}  # only 1 solution globally
        elif custom_conf['query'] == 'BA':
            gq = nx.barabasi_albert_graph(n=custom_conf['query_n'],
                                          m=custom_conf['query_m'],
                                          seed=0)
            gq = nx.convert_node_labels_to_integers(gq)
            gt_new = nx.disjoint_union(gt, gq)
            assert gt_new.number_of_nodes() == gt.number_of_nodes() + gq.number_of_nodes()
            assert gt_new.number_of_edges() == gt.number_of_edges() + gq.number_of_edges()
            gt_new.add_edge(len(gt) - 1, len(gt))
            true_nn_map = {}
            for i in range(len(gq)):
                true_nn_map[i] = i + len(gt)
            gt = gt_new
        else:
            raise NotImplementedError()
    elif custom_conf['target'] == 'wheel':
        gt = nx.convert_node_labels_to_integers(
            wheel_graph(custom_conf['n']))
        gq = _gen_triangle()
        true_nn_map = None  # too many solutions
    else:
        raise NotImplementedError()

    if custom_conf['turn_off_true_nn_map']:
        true_nn_map = None

    num_pairs = custom_conf['num_pairs']
    name = f'{_flatten_dict_to_str(custom_conf)}'

    for i in tqdm(range(num_pairs)):
        assert False, 'deprecated'
        gp, gq, gt_filtered = _gen_load_CS_gp_filter(
            gq, gt, true_nn_map=true_nn_map, ppn=None, enc=None)

        saver.log_info(f'\n\n--- Query {i}/{num_pairs}: {name}')
        print_g(gq.nx_graph, 'gq', print_func=saver.log_info)
        print_g(gt_filtered.nx_graph, 'gt', print_func=saver.log_info)

        saver.save_graph_as_gexf(gt, f'{name}_pair{i}_target.gexf')
        saver.save_graph_as_gexf(gt_filtered.nx_graph,
                                 f'{name}_pair{i}_targetfiltered.gexf')
        saver.save_graph_as_gexf(gq.nx_graph, f'{name}_pair{i}_query.gexf')

        # saver.save_graph_as_gexf(_change_label(gq), f'{basename(qpn)}.gexf')
        yield gp


def _flatten_dict_to_str(d):
    return ', '.join("{!s}={!r}".format(key, val) for (key, val) in d.items())


def _gen_triangle():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(0, 2)
    return G


def _gen_hexagon():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(5, 0)
    return G


def _gen_heptagon():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(5, 6)
    G.add_edge(6, 0)
    return G


if __name__ == '__main__':
    dl = get_data_loader_wrapper('train')
    for gp in dl:
        print(gp)
        # OurGraphPair(gq, gto)
    exit()

    task = 'visualize'

    from daf import get_CS
    from collections import defaultdict
    from saver import saver
    import time

    # dataset_name_li = ['imsm-human_unlabeled', 'imsm-dblp_unlabeled', 'imsm-hprd_unlabeled', 'imsm-youtube_unlabeled']
    # subgroup_li = ['dense_32', 'dense_64', 'dense_128']
    dataset_name_li = ['imsm-human_unlabeled', 'imsm-dblp_unlabeled', 'imsm-hprd_unlabeled', 'imsm-youtube_unlabeled']
    subgroup_li = ['dense_128']
    dataset_name2subgroup2time = defaultdict(lambda: defaultdict(float))
    for dataset_name in dataset_name_li:
        for subgroup in subgroup_li:
            if task == 'check-isomorphism':
                path_indices1 = [0,50]
                path_indices2 = [100,200]
                dl1 = get_data_loader(dataset_name, None, subgroup, path_indices1, sample_size=None)
                dl2 = get_data_loader(dataset_name, None, subgroup, path_indices2, sample_size=None)
                for i, gp1 in enumerate(dl1):
                    for j, gp2 in enumerate(dl2):
                        gq1 = gp1.gq.nx_graph
                        gq2 = gp2.gq.nx_graph
                        if nx.faster_could_be_isomorphic(gq1, gq2):
                            print(f'{dataset_name} {subgroup}: ({i}, {j}) may be problematic!')
                            exit(-1)
            elif task == 'check-CS-generation-time':
                CS_time = 0
                path_indices = [0,10]
                dl = get_data_loader(dataset_name, None, subgroup, path_indices, sample_size=None)

                qgraphs_paths, tpn, pkl_folder = _get_imsm_paths(dataset_name, subgroup, path_indices)
                for qpn in qgraphs_paths:
                    t0 = time.time()
                    get_CS(qpn, tpn)
                    CS_time += (time.time() - t0)
                CS_time /= float(max(path_indices) - min(path_indices))
                dataset_name2subgroup2time[dataset_name][subgroup] = CS_time
                print(f'dataset {dataset_name} {subgroup} took {CS_time} sec to generate CS')
            elif task == 'visualize':
                CS_time = 0
                path_indices = [0,1]
                dl = get_data_loader(dataset_name, None, subgroup, path_indices, sample_size=None)

                qgraphs_paths, tpn, pkl_folder = _get_imsm_paths(dataset_name, subgroup, path_indices)
                gt = read_tve(tpn)
                print(f'writing {dataset_name}-{subgroup} target graph...')
                nx.write_gexf(gt, join('tuning', f'{dataset_name}-{subgroup}-t.gexf'))
                print('done!')
                for i, qpn in enumerate(tqdm(qgraphs_paths)):
                    gq = read_tve(qpn)
                    nx.write_gexf(gq, join('tuning', f'{dataset_name}-{subgroup}-q{i}.gexf'))
            else:
                pass

    if task == 'check-isomorphism':
        print('no leakage for datasets:', ', '.join(dataset_name_li))
    elif task == 'check-CS-generation-time':
        saver.log_info(dataset_name2subgroup2time)
        saver.log_info('-----------')
        for dataset_name, subgroup2time in dataset_name2subgroup2time.items():
            for subgroup, CS_time in subgroup2time.items():
                saver.log_info(f'dataset={dataset_name}={subgroup}=took={CS_time}=sec to generate CS')
    else:
        pass
