from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from os.path import join

from NSUBS.model.OurSGM.saver import saver


class EmbeddingLogger():
    def __init__(self):
        self.search_iter = ''
        self.buffer_entry_id = ''
        self.train_iter = ''
        self.action = ''
        self.action_best_li = ''
        self.tensorboard_logged = False

    def get_pn_state(self):
        fn = f'regularization_search_itr_{self.search_iter}_buffer_entry_{self.buffer_entry_id}_train_itr_{self.train_iter}.png'
        pn = join(saver.get_plot_dir(), fn)
        return pn

    def get_pn_action(self):
        good_action = 'good' if self.action in self.action_best_li else 'bad'
        fn = f'V_search_itr_{self.search_iter}_buffer_entry_{self.buffer_entry_id}_train_itr_{self.train_iter}_action_{self.action}_{good_action}.png'
        pn = join(saver.get_plot_dir(), fn)
        return pn

    def plot_embeddings(self, X_q, X_t, g_q, g_t, nn_map, candidate_map, override=None, filter_g_t=True,
             pn=None):
        print('tsne plotting start')
        override = {'g_q': {}, 'g_t': {}} if override is None else override
        if filter_g_t:
            g_t, nn_map, candidate_map, X_t, override = \
                self.filter_g_t_for_logging(g_t, nn_map, candidate_map, X_t, override)
            print('g_t filtering done')
        else:
            g_t = nx.subgraph(g_t, set().union(*[set(v_li) for v_li in candidate_map.values()],
                                               nn_map.values()))
        # tsne = TSNE(2)
        # print(g_q.number_of_nodes(), g_t.number_of_nodes(), len(nn_map), len(candidate_map), X_q.shape, X_t.shape, len(override))
        tsne = TSNE()  # n_jobs=1)
        print('tsne start')
        # print(torch.cat([X_q, X_t], dim=0).detach().cpu().numpy())
        # print('ducks')
        tsne_result = tsne.fit_transform(torch.cat([X_q, X_t], dim=0).detach().cpu().numpy())
        print('tsne done')

        X_q_tsne = tsne_result[:X_q.shape[0]]
        X_t_tsne = tsne_result[X_q.shape[0]:]
        cmap = self._get_cmap(nn_map, candidate_map)
        g_t = self._get_g_t_filtered(g_t, nn_map, candidate_map)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        print(f'plot metdata collected, |g_q| = {g_q.number_of_nodes()}')

        self._plot_state_single_graph(
            g_q, X_q_tsne, cmap['g_q'], nn_map.keys(), ax, override['g_q'], marker='x', alpha=0.8,
            core_size=50, default_size=50, plot_edges=True)
        print(f'Xq plotted, |gt| = {g_t.number_of_nodes()}')
        self._plot_state_single_graph(
            g_t, X_t_tsne, cmap['g_t'], nn_map.values(), ax, override['g_t'], marker='o')
        print('Xt plotted')

        if pn is None:
            plt.show()
        else:
            plt.savefig(pn)
            saver.log_info(f'EmbeddingLogger: saved plot to {pn}')

        # exit(-1)
        plt.close()

    def filter_g_t_for_logging(self, g_t, nn_map, candidate_map, X_t, override):
        nn_map.values()
        frontier_nids = set().union(*[set(g_t.neighbors(nid)) for nid in nn_map.values()])
        candidate_nids = set().union(*[set(v_li) for v_li in candidate_map.values()])
        valid_nids = list((frontier_nids.intersection(candidate_nids)).union(nn_map.values()))
        relabel_dict = {nid: i for (i, nid) in enumerate(valid_nids)}
        sg = nx.subgraph(g_t, valid_nids)
        sg = nx.relabel_nodes(sg, relabel_dict)
        override['g_t'] = \
            {relabel_dict[nid]: val for (nid, val) in override['g_t'].items() if
             nid in relabel_dict}
        nn_map_new = {nid_k: relabel_dict[nid_v] for (nid_k, nid_v) in nn_map.items()}
        candidate_map_new = {
        nid_k: [relabel_dict[nid_v] for nid_v in nid_v_li if nid_v in relabel_dict] for
        (nid_k, nid_v_li) in candidate_map.items()}
        return sg, nn_map_new, candidate_map_new, X_t[valid_nids], override

    def _get_g_t_filtered(self, g_t, nn_map, candidate_map):
        return nx.subgraph(g_t, set().union(*[set(v_li) for v_li in candidate_map.values()],
                                            nn_map.values()))

    def _get_cmap(self, nn_map, candidate_map):
        cmap = {'g_q': {}, 'g_t': {}}
        # colors_cand_map = [(0.5, 0.5, 0.5, 1)]*len(candidate_map)
        colors_cand_map = list(cm.rainbow(np.linspace(0, 1, len(candidate_map))))
        # ListedColormap(cm.get_cmap('hsv', len(candidate_map))(np.linspace(0.3, 0.7, len(candidate_map)))).colors
        # cm.rainbow(np.linspace(0, 1, len(candidate_map))))
        for i, (u, v_li) in enumerate(candidate_map.items()):
            cmap['g_q'][u] = colors_cand_map[i]
            for v in v_li:
                cmap['g_t'][v] = colors_cand_map[i]
        colors_nn_map = [(0, 0, 0, 1)] * len(nn_map)
        # colors_nn_map = cm.get_cmap('inferno', len(nn_map)).colors
        for i, (u, v) in enumerate(nn_map.items()):
            cmap['g_q'][u] = np.array(colors_nn_map[i])
            cmap['g_t'][v] = np.array(colors_nn_map[i])
        return cmap

    def _plot_state_single_graph(self, g, X, cmap, core, ax, override, marker='.', alpha=0.5,
                                 core_size=10, default_size=5, plot_edges=False):
        if plot_edges:
            for nid1, nid2 in g.edges():
                X_edge = np.stack([X[nid1], X[nid2]], axis=-1)
                ax.plot(X_edge[0], X_edge[1], color='gray', zorder=1, linewidth=0.1)
        gnodes = list(g.nodes())
        X_node = np.stack([X[nid] for nid in gnodes], axis=-1)
        nid_li = list(override.keys())
        ax.scatter(
            X_node[0][nid_li], X_node[1][nid_li],
            color=[override[nid]['color'] for nid in nid_li],
            # np.stack([override[nid]['color'] for nid in nid_li], axis=0),
            zorder=2,
            alpha=0.2,
            s=[override[nid]['size'] for nid in nid_li]
        )
        ax.scatter(
            X_node[0], X_node[1],
            color=np.stack([cmap[nid] for nid in gnodes], axis=0),
            zorder=2,
            alpha=alpha,
            marker=marker,
            s=[(core_size if nid in core else default_size) for nid in gnodes]
        )

    def log_embeddings_tensorboard(self, X_q, X_t):
        if self.tensorboard_logged:
            return
        X = torch.cat([X_t, X_q], dim=0)
        # print('@@@', X_t.shape)
        tnum_nodes = X_t.shape[0]
        tnames = [f't_{i}' for i in range(tnum_nodes)]
        qnum_nodes = X_q.shape[0]
        qnames = [f'q_{i}' for i in range(qnum_nodes)]
        # exit(-1)
        saver.writer.add_embedding(X, metadata=tnames + qnames, label_img=None,
                                   tag=f'node_embeddings_{self.search_iter}')
        saver.log_info(f'Embedding logger: run tensorboard for '
                       f'{self.search_iter} to see embeddings of shape {X.shape}')
        self.tensorboard_logged = True


from networkx.generators.random_graphs import barabasi_albert_graph

if __name__ == '__main__':
    D = 32

    g_q = barabasi_albert_graph(32, 3)
    g_t = barabasi_albert_graph(1000, 3)
    X_q = torch.randn((g_q.number_of_nodes(), D))
    X_t = torch.randn((g_t.number_of_nodes(), D))


    def get_nid_li(i, n_itr, n_max):
        li = []
        while i < n_max:
            li.append(i)
            i += n_itr
        return li


    nn_map = {i: i for i in range(int(g_q.number_of_nodes() / 2))}
    candidate_map = {i: get_nid_li(i, i + int(g_q.number_of_nodes() / 2), g_t.number_of_nodes()) for
                     i in range(int(g_q.number_of_nodes() / 2), g_q.number_of_nodes())}

    nn_map_keys = list(nn_map.keys())
    X_t[[nn_map[k] for k in nn_map_keys]] = X_q[nn_map_keys]

    override = \
        {
            'g_q': {
                'color': {5: (0, 1, 0, 1)},
                'size': {5: 300}
            },
            'g_t': {
                'color': {5: (1, 0, 0, 1)},
                'size': {5: 300}
            }
        }

    logger = EmbeddingLogger()
    logger.plot(X_q, X_t, g_q, g_t, nn_map, candidate_map, filter_g_t=False)  # override=override)
    print('done!')
