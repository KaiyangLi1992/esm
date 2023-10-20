from NSUBS.model.OurSGM.utils_our import pyg_from_networkx
from NSUBS.src.utils import OurTimer, save_pickle, load_pickle, print_stats
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std
from NSUBS.model.OurSGM.config import FLAGS

import torch

class OurGraphPair():
    def __init__(self, gq, gt, CS, daf_path_weights, true_nn_map=None, encoder=None, relabel_dict=None):
        if type(gt) is not OurGraph:
            gt = OurGraph(gt, encoder=encoder)
        if type(gq) is not OurGraph:
            gq = OurGraph(gq, encoder=encoder)
        self.gq = gq
        self.gt = gt
        self.CS = CS
        self.daf_path_weights = daf_path_weights
        self.true_nn_map = true_nn_map
        self.relabel_dict = relabel_dict
        assert gt.x.shape[1] == gq.x.shape[1], f'{gt.x.shape} and {gq.x.shape}'

    def get_d_in_raw(self):
        return self.gt.x.shape[1]

    def to(self, device):
        self.gq.to(device)
        self.gt.to(device)


class OurGraph():
    def __init__(self, g, encoder=None):
        self.nx_graph = g
        data = pyg_from_networkx(g)
        self.edge_index = data.edge_index  # torch.LongTensor(list(g.edges)).t().contiguous()

        # self.x = torch.ones((g.number_of_nodes(), 1), dtype=torch.float)

        if encoder is None:
            # TODO: use local degree profile and check node ordering!!! (0-indexed? sorted?)
            self.x = torch.ones((g.number_of_nodes(), 1), dtype=torch.float)
        else:
            X = []
            for node, ndata in sorted(g.nodes(data=True)):
                X.append([ndata['label']])
            X = encoder.transform(X).toarray()
            self.x = torch.tensor(X, dtype=torch.float)
            # assert self.x.shape == (g.number_of_nodes(), encoder.n_values_[0])
            assert self.x.shape == (g.number_of_nodes(), len(encoder.categories_[0]))

        print('before ldf:',self.x.shape)
        if FLAGS.append_ldf:
            self.x = torch.cat((self.x, self.get_ldf()), dim=1)
        print('after ldf:',self.x.shape)

        self.x_encoded = None

    # def update_sel_nodes(self, sel_nodes, in_place=True):
    #     sel_nodes_intersection = self.sel_nodes.intersection(sel_nodes)
    #     sel_g = nx.subgraph(self.nx_graph, sel_nodes_intersection)
    #     sel_g_edges = [[int(x) for x in lst] for lst in sel_g.edges()]
    #     edge_index = torch.LongTensor(sel_g_edges).t().contiguous().to(FLAGS.device)
    #     if in_place:
    #         self.sel_nodes = sel_nodes
    #         self.edge_index = edge_index
    #     return sel_nodes, edge_index

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.x = self.x.to(device)

    def get_ldf(self):
        row, col = self.edge_index
        N = self.nx_graph.number_of_nodes()

        deg = torch.tensor([val for (node, val) in sorted(self.nx_graph.degree(), key=lambda pair: pair[0])], dtype=torch.float)
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        X = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        return X


import networkx as nx


class SRW_RWF_ISRW:

    def __init__(self, seed):
        self.growth_size = -1
        self.T = 100  # number of iterations
        # with a probability (1-fly_back_prob) select a neighbor node
        # with a probability fly_back_prob go back to the initial vertex
        self.fly_back_prob = 0.15
        self.seed = seed


    def random_walk_induced_graph_sampling(self, complete_graph, nodes_to_sample):
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        while True:
            rtn = self._try_sampling(complete_graph, nodes_to_sample)
            if rtn is not None:
                return rtn


    def _try_sampling(self, complete_graph, nodes_to_sample):
        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = self.seed.randint(0, nr_nodes - 1)

        Sampled_nodes = {complete_graph.nodes[index_of_first_random_node]['id']}

        iteration = 1
        # nodes_before_t_iter = 0
        curr_node = index_of_first_random_node
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            if len(edges) == 0:
                return None
            index_of_edge = self.seed.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            # if iteration % self.T == 0:
            #     if ((len(Sampled_nodes) - nodes_before_t_iter) < self.growth_size):
            #         curr_node = self.seed.randint(0, nr_nodes - 1)
            #     nodes_before_t_iter = len(Sampled_nodes)

            if iteration == 10000:
                return None

        sampled_graph = complete_graph.subgraph(Sampled_nodes)

        return sampled_graph



def xtract(g):
    # return max(nx.connected_component_subgraphs(g), key=len)
    li = [g.subgraph(c) for c in nx.connected_components(g)]
    return max(li, key=len)



def random_walk_sampling(g, max_len, srw_rwf_isrw, check_connected=True):
    if check_connected:
        assert nx.is_connected(g)
    subg = srw_rwf_isrw.random_walk_induced_graph_sampling(g, max_len)
    # subg = xtract(subg)
    assert nx.is_connected(subg)
    assert len(subg) == max_len
    return subg



from networkx import relabel_nodes

def convert_node_labels_to_integers_with_mapping(G, first_label=0, ordering="default",
                                    label_attribute=None):
    N = G.number_of_nodes() + first_label
    if ordering == "default":
        mapping = dict(zip(G.nodes(), range(first_label, N)))
    elif ordering == "sorted":
        nlist = sorted(G.nodes())
        mapping = dict(zip(nlist, range(first_label, N)))
    elif ordering == "increasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    elif ordering == "decreasing degree":
        dv_pairs = [(d, n) for (n, d) in G.degree()]
        dv_pairs.sort()  # in-place sort from lowest to highest degree
        dv_pairs.reverse()
        mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
    else:
        raise nx.NetworkXError('Unknown node ordering: %s' % ordering)

    H = relabel_nodes(G, mapping)
    # create node attribute with the old label
    if label_attribute is not None:
        nx.set_node_attributes(H, {v: k for k, v in mapping.items()},
                               label_attribute)
    rev_mapping = _reverse_map(mapping)
    assert len(rev_mapping) == len(mapping)
    return H, rev_mapping # maps new graph node ids to original


def _reverse_map(m):
    return {v: k for k, v in m.items()}