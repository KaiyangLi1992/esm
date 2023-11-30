from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.src.utils import get_ts, create_dir_if_not_exists, save
from NSUBS.model.OurSGM.utils_our import get_our_dir, get_model_info_as_str, \
    get_model_info_as_command, extract_config_code, plot_scatter_line, plot_dist, get_flag
from tensorboardX import SummaryWriter
from collections import OrderedDict
from pprint import pprint
from os.path import join, basename
from NSUBS.src.utils import get_host
import torch
import networkx as nx
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
import os

class ParameterSaver:
    def __init__(self, log_dir='log'):
        self.log_dir = log_dir
        # 创建日志文件夹
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def save(self, params, file_name='params.log'):
        file_path = os.path.join(self.log_dir, file_name)
        # 写入参数
        with open(file_path, 'a') as file:
            for key, value in params.items():
                file.write(f'{key}: {value}\n')
            # 添加一个换行符来分隔不同的保存点
            file.write('\n')




class Saver(object):
    def __init__(self):
        model_str = self._get_model_str()
        self.logdir = join(
            get_our_dir(),
            'logs',
            '{}_{}'.format(model_str, get_ts()))  # get current unique time stamp
        create_dir_if_not_exists(self.logdir)
        print(f'Logging to {self.logdir}')
        self.writer = SummaryWriter(self.logdir)
        self.model_info_f = self._open('model_info.txt')
        self.plotdir = join(self.logdir, 'plot')
        self.objdir = join(self.logdir, 'obj')
        self._log_model_info()
        self._save_conf_code()
        # print('Logging to {}'.format(self.logdir))

    def _open(self, f):
        return open(join(self.logdir, f), 'w')

    def close(self):
        self.writer.close()
        self.log_info(f'{self.logdir}')
        self.log_info(f'{basename(self.logdir)}')
        if hasattr(self, 'log_f'):
            self.log_f.close()
        if hasattr(self, 'results_f'):
            self.results_f.close()

    def get_log_dir(self):
        return self.logdir

    def get_plot_dir(self):
        create_dir_if_not_exists(self.plotdir)
        return self.plotdir

    def get_obj_dir(self):
        create_dir_if_not_exists(self.objdir)
        return self.objdir

    def save_as_pickle(self, d, name):
        p = join(self.get_obj_dir(), name)
        save(d, p, print_msg=True, use_klepto=False)

    def log_list_of_lists_to_csv(self, lol, fn, delimiter=','):
        import csv
        fp = open(join(self.logdir, fn), 'w+')
        csv_writer = csv.writer(fp, delimiter=delimiter)
        for l in lol:
            csv_writer.writerow(l)
        fp.close()

    def log_nxgraph(self, g, fn, silent=False, gexf=False):
        pn = join(self.get_obj_dir(), fn)
        if not silent:
            print('saving...')
        t0 = time.time()
        if gexf:
            nx.write_gexf(g, pn)
        else:
            nx.write_gpickle(g, pn)
        elapsed_time = time.time() - t0
        if not silent:
            print(f'saving done! elapsed time: {elapsed_time}')

    def log_obj(self, obj, fn):
        with open(join(self.get_obj_dir(), fn), 'wb') as handle:
            pickle.dump(obj, handle, pickle.HIGHEST_PROTOCOL)

    def log_dict_to_json(self, dictionary, fn):
        import json

        # as requested in comment
        with open(join(self.get_obj_dir(), fn), 'w') as file:
            file.write(json.dumps(dictionary))

    def log_model_architecture(self, model, tag):
        print(tag)
        print(model)
        self.model_info_f.write('{}:\n'.format(tag))
        self.model_info_f.write('{}\n'.format(model))
        self.model_info_f.write('\n\nDetails:')
        for name, module in model.named_modules():
            self.model_info_f.write('{}\n'.format(name))
            self.model_info_f.write('{}\n\n'.format(module))
        # self.model_info_f.close()  # TODO: check if future if we write more to it

    def log_info(self, s, silent=False):
        if not silent:
            print(s)
        if not hasattr(self, 'log_f'):
            self.log_f = self._open('log.txt')
        self.log_f.write('{}\n'.format(s))

    def log_info_new_file(self, s, fn):
        # print(s)
        log_f = open(join(self.logdir, fn), 'a')
        log_f.write('{}\n'.format(s))
        log_f.close()

    def _save_conf_code(self):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(extract_config_code())
        p = join(self.get_log_dir(), 'FLAGS')
        save({'FLAGS': FLAGS}, p, print_msg=False)

    # def save_flags(self, fn):
    #     p = join(self.get_log_dir(), fn)
    #     save({'FLAGS': FLAGS}, p, print_msg=False)

    def get_model_dir(self):
        model_dir = join(self.logdir, 'models')
        create_dir_if_not_exists(model_dir)
        return model_dir

    def save_trained_model(self, trained_model, ext=''):
        ext = '_' + ext if ext != '' else ext
        p = join(self.get_model_dir(), 'trained_model{}.pt'.format(ext))
        torch.save(trained_model.state_dict(), p)
        saver.log_info('Trained model saved to {}'.format(p))

    def log_bd_stats(self, bd_stats_dict):
        for pair_id, bd_stats_list in bd_stats_dict.items():
            for iteration, bd_stats in enumerate(bd_stats_list):
                bd_stats_connected_arr = self.flatten_list_tuple_into_np_arr(
                    bd_stats.bd_stats_connected)
                bd_stats_unconnected_arr = self.flatten_list_tuple_into_np_arr(
                    bd_stats.bd_stats_unconnected)
                if len(bd_stats_connected_arr) > 1:
                    plot_dist(
                        bd_stats_connected_arr, f'{pair_id}_{iteration}_conn', self.get_plot_dir(),
                        saver=self)
                if len(bd_stats_unconnected_arr) > 1:
                    plot_dist(
                        bd_stats_unconnected_arr, f'{pair_id}_{iteration}_unconn',
                        self.get_plot_dir(), saver=self)

    def flatten_list_tuple_into_np_arr(self, li):
        li_flatten = []
        for elt in li:
            for elt_elt in elt:
                li_flatten.append(elt_elt)
        return np.array(li_flatten)

    def log_scatter_mcs(self, cur_id, iter, result_d):
        sp = join(self.get_obj_dir(), f'{iter}_val')
        save(result_d, sp)
        self._save_to_result_file(f'iter {iter} val result', to_print=True)
        for label, data_dict in result_d.items():
            if FLAGS.val_debug:
                g1, g2 = data_dict['g1'], data_dict['g2']
                nx.write_gexf(g1, join(self.get_obj_dir(), f'{cur_id}_{g1.graph["gid"]}.gexf'))
                nx.write_gexf(g2, join(self.get_obj_dir(), f'{cur_id}_{g2.graph["gid"]}.gexf'))
            plot_scatter_line(data_dict['result'], label, self.get_plot_dir())

        for label, data_dict in result_d.items():
            for model_name, data_dict_elt in data_dict['result'].items():
                incumbent_size_list = data_dict_elt['incumbent_data']
                runtime = data_dict_elt['runtime']
                self._save_to_result_file(f'  num_iters={len(incumbent_size_list)}, '
                                          f'runtime={runtime}, '
                                          f'mcs={incumbent_size_list[-1]}, '
                                          f'method={model_name}', to_print=True)

    def save_graph_as_gexf(self, g, fn):
        nx.write_gexf(g, join(self.get_obj_dir(), fn))

    def save_eval_result_dict(self, result_dict, label):
        self._save_to_result_file(label)
        self._save_to_result_file(result_dict)

    def save_pairs_with_results(self, test_data, train_data, info):
        p = join(self.get_log_dir(), '{}_pairs'.format(info))
        save({'test_data_pairs':
                  self._shrink_space_pairs(test_data.dataset.pairs),
              # 'train_data_pairs':
              # self._shrink_space_pairs(train_data.dataset.pairs)
              },
             p, print_msg=False)

    def save_ranking_mat(self, true_m, pred_m, info):
        p = join(self.get_log_dir(), '{}_ranking_mats'.format(info))
        save({'true_m': true_m.__dict__, 'pred_m': pred_m.__dict__},
             p, print_msg=False)

    def save_global_eval_result_dict(self, global_result_dict):
        p = join(self.get_log_dir(), 'global_result_dict')
        save(global_result_dict, p, print_msg=False)

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    # def clean_up_saved_models(self, best_iter):
    #     for file in glob('{}/models/*'.format(self.get_log_dir())):
    #         if str(best_iter) not in file:
    #             system('rm -rf {}'.format(file))

    def save_exception_msg(self, msg):
        with self._open('exception.txt') as f:
            f.write('{}\n'.format(msg))

    def _get_model_str(self):
        def _check(s, f):
            if f in ['do_train', 'do_validation', 'do_test']:
                if s == True:
                    if f == 'do_train':
                        return 'train'
                    elif f == 'do_validation':
                        return 'val'
                    else:
                        assert f == 'do_test'
                        return 'test'
                else:
                    return None
            else:
                return s

        li = []
        key_flags = ['model', 'do_train', 'do_validation', 'do_test',
                     'train_dataset', 'train_subgroup',
                     'test_dataset', 'test_subgroup', 'matching_order', 'timeout']
        for f in key_flags:
            s = get_flag(f)
            s = _check(s, f)
            if s is not None:
                li.append(str(s))
        return '_'.join(li)

    def _log_model_info(self):
        s = get_model_info_as_str()
        print(s)
        c = get_model_info_as_command()
        self.model_info_f.write(s)
        self.model_info_f.write('\n\n')
        self.model_info_f.write(c)
        self.model_info_f.write('\n\n')
        self.writer.add_text('model_info_str', s)
        self.writer.add_text('model_info_command', c)
        self.model_info_f.flush()
        # exit(-1)

    def log_new_FLAGS_to_model_info(self):
        self.model_info_f.write('----- new model info after loading\n')
        self._log_model_info()

    def _save_to_result_file(self, obj, name=None, to_print=False):
        if not hasattr(self, 'results_f'):
            self.results_f = self._open('results.txt')
        if type(obj) is dict or type(obj) is OrderedDict:
            # self.f.write('{}:\n'.format(name))
            # for key, value in obj.items():
            #     self.f.write('\t{}: {}\n'.format(key, value))
            pprint(obj, stream=self.results_f)
        elif type(obj) is str:
            if to_print:
                print(obj)
            self.results_f.write('{}\n'.format(obj))
        else:
            self.results_f.write('{}: {}\n'.format(name, obj))
        self.results_f.flush()

    def _shrink_space_pairs(self, pairs):
        for _, pair in pairs.items():
            # print(pair.__dict__)
            pair.shrink_space_for_save()
            # pass
            # print(pair.__dict__)
            # exit(-1)
        return pairs

    def add_mcs_result(self, mcs_result):
        if not hasattr(self, 'mcs_result_li'):
            self.mcs_result_li = []
        self.mcs_result_li.append(mcs_result)

    def plot_sequence(self, x, y, ttl='', xlab='', ylab=''):
        plt.plot(x, y)
        plt.title(ttl)
        plt.xlabel(xlab)
        plt.xlabel(ylab)
        plt.savefig(join(self.get_log_dir(), f'{ttl}.png'), bbox_inches='tight')
        plt.close()

    def plot_search_tree(self, smts, label):
        saver.log_info(f'Plotting search tree of {len(smts.search_tree.nxgraph)} nodes')
        # if not FLAGS.MCTS:
        #     assert smts.search_tree.nxgraph.number_of_nodes() == smts.num_iters + 1, \
        #         f'{smts.search_tree.nxgraph.number_of_nodes()} {smts.num_iters}'

        if not hasattr(self, 'tree_dir'):
            self.tree_dir = join(self.logdir, 'tree')
            create_dir_if_not_exists(self.tree_dir)

        # def tree_to_newick(g, root=None, node_attrib=None):
        #     if root is None:
        #         roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        #         assert 1 == len(roots)
        #         root = roots[0][0]
        #     subgs = []
        #     for child in g[root]:
        #         if len(g[child]) > 0:
        #             subgs.append(tree_to_newick(g, root=child, node_attrib=node_attrib))
        #         else:
        #             if node_attrib is not None:
        #                 subgs.append(f'{child}:{g.nodes[child][node_attrib]}')
        #             else:
        #                 subgs.append(str(child))
        #     # if node_attrib is not None:
        #     #     subgs = [f'{s}:{g.nodes[s][node_attrib]}' for s in subgs]
        #     # else:
        #     # subgs = [str(s) for s in subgs]
        #     return "(" + ','.join(subgs) + ")"
        #
        # s = tree_to_newick(smts.tree, root=0, node_attrib='num_solutions') + ';'

        G = smts.search_tree.nxgraph
        if G.number_of_nodes() < 1000 and get_host() == 'anonymous': # doesn't work on server, graphviz pacakge messed up
            import ete3
            from ete3 import Tree, TreeStyle
            root = 0

            # subtrees = {node: ete3.Tree(name=node) for node in G.nodes()}
            subtrees = {}
            for node, nd in G.nodes(data=True):
                tnode = ete3.Tree(name=f'n{node}')
                u = nd.get('u', '-')
                v = nd.get('v', '-')
                vscore = nd.get('vscore', '-')
                vli_size = nd.get('vli_size', '-')
                global_num_solutions = nd.get('global_num_solutions', '-')
                depth = nd.get('depth', '-')
                tnode.add_features(depth=f'd={depth}',
                                   num_solutions=f'sol={global_num_solutions}',
                                   vli_size=f'vs={vli_size}',
                                   uv=f'uv=({u},{v})(_,{vscore}). ')

                # d = nd['depth']
                # x = nd['num_solutions']
                # uv = nd['uv']
                # tnode.add_features(depth=f'd={d}', num_solutions=f's={x}', uv=f'uv=({uv})')
                subtrees[node] = tnode

            # subtrees = {node: ete3.Tree(name=node) for node in G.nodes()}
            [*map(lambda edge: subtrees[edge[0]].add_child(subtrees[edge[1]]), G.edges())]
            ete3_tree = subtrees[root]

            with open(join(self.tree_dir, f'{label}.txt'), 'w') as f:
                f.write(ete3_tree.get_ascii(
                    attributes=['name', 'depth', 'num_solutions', 'vli_size', 'uv']))

            # for node, nd in G.nodes(data=True):
            #     tnode = ete3_tree.search_nodes(name=node)[0]
            #     # print(tnode)
            #     tnode.add_features(num_solutions=nd['num_solutions'])

            # ete3_tree = Tree(s)

            # print(ete3_tree.get_ascii())

            # exit()

            # print(1)
            if len(G) > 100009:
                saver.log_info(f'Skip large tree {label}')
                return
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            plt.figure()
            # print(2)
            nx.draw(G, pos=pos, node_size=1, width=0.1, arrows=False)
            # print(3)
            # print(4)
            plt.savefig(join(self.tree_dir, f'{label}.png'), bbox_inches='tight')
            # print(5)

            # ts = TreeStyle()
            # ts.show_leaf_name = True
            # ts.rotation = 90

            # ete3_tree.render(join(self.tree_dir, f'{label}.png'))

            # nx.write_gexf(smts.tree, join(self.tree_dir, f'{label}.gexf'))

            saver.log_info(f'Tree plotted {label} with {G.number_of_nodes()} nodes')
            plt.close()

        for nid in G.nodes:
            if G.nodes[nid]['u'] is None:
                G.nodes[nid]['u'] = -1
            # G.nodes[nid]['P_li_max'] = -1 if len(G.nodes[nid]['P_li']) == 0 else float(np.array(G.nodes[nid]['P_li']).max())
            # G.nodes[nid]['P_li_mean'] = -1 if len(G.nodes[nid]['P_li']) == 0 else float(np.array(G.nodes[nid]['P_li']).mean())
            # G.nodes[nid]['P_li_min'] = -1 if len(G.nodes[nid]['P_li']) == 0 else float(np.array(G.nodes[nid]['P_li']).min())
            # G.nodes[nid]['P_li'] = '_'.join([str(p) for p in G.nodes[nid]['P_li']])
            G.nodes[nid]['out_policy'] = '_'.join([str(nid) for nid in G.nodes[nid]['out_policy']])
            # if float(G.nodes[nid]['out_value']) > 1000:
            #     print(';@*@&&@', float(G.nodes[nid]['out_value']))
            #     print(G.nodes[nid]['out_value'])
            #     exit()
            # del G.nodes[nid]['bilin_emb']
            # del G.nodes[nid]['g_emb']
            G.nodes[nid]['out_value'] = float(G.nodes[nid]['out_value'])
            G.nodes[nid]['pi_sm_normalized_valid'] = '_'.join([str(x) for x in G.nodes[nid]['pi_sm_normalized_valid']])
        for eid in G.edges:
            G.edges[eid]['action'] = '_'.join([str(nid) for nid in G.edges[eid]['action']])
        nx.write_gexf(G, join(self.tree_dir, f'{label}.gexf'))

    def plot_solutions(self, smts, gq, gt, label, true_nn_map, CS=None):
        if not hasattr(self, 'sol_dir'):
            self.sol_dir = join(self.logdir, 'solution')
            create_dir_if_not_exists(self.sol_dir)

        def _change_label(g):
            for node, ndata in g.nodes(data=True):
                nx.set_node_attributes(g, {node: ndata.get('label', 0)},
                                       name='type')  # gephi does not plot label well

            return g

        gq = gq.nx_graph.copy()
        gt = gt.nx_graph.copy()

        gq = _change_label(gq)
        gt = _change_label(gt)

        for i, nn_map in enumerate(smts.solutions):
            aname = f'sol_{i}'
            for node1, node2 in nn_map.items():
                nx.set_node_attributes(gq, {node1: 1}, name=aname)
                nx.set_node_attributes(gt, {node2: 1}, name=aname)

        if true_nn_map is not None:
            # print('*****')
            aname = f'sol_true'
            # print('@@@@@ true_nn_map', len(true_nn_map), true_nn_map.values())
            for node1, node2 in true_nn_map.items():
                nx.set_node_attributes(gq, {node1: 1}, name=aname)
                nx.set_node_attributes(gt, {node2: 1}, name=aname)
                if CS is not None and node2 not in CS[0][node1]:
                    raise ValueError(f'{node1} in G1 has true nn map {node2} '
                                     f'but it is not in CS of {node1}: {CS[0][node1]}')

        if CS is not None:
            for node1, node2s in CS[0].items():
                for node2 in node2s:
                    if node2 in true_nn_map.values():
                        nlabel = 2
                    else:
                        nlabel = 1
                    nx.set_node_attributes(gt, {node2: nlabel}, name=f'CS_{node1}')

        nx.write_gexf(gq, join(self.sol_dir, f'{label}_q.gexf'))
        nx.write_gexf(gt, join(self.sol_dir, f'{label}_t.gexf'))

        saver.log_info(f'{len(smts.solutions)} solutions plotted {label}')

    def save_search(self, smts, label):
        if not hasattr(self, 'smts_dir'):
            self.smts_dir = join(self.logdir, 'smts')
            create_dir_if_not_exists(self.smts_dir)

        save({'smts': smts}, join(self.smts_dir, f'{label}_smts'),
             print_msg=True, use_klepto=False)


saver = Saver()  # can be used by `from saver import saver`
