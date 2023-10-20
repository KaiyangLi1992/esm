from NSUBS.model.OurSGM.utils_our import get_flags_with_prefix_as_list
from torch_geometric.transforms import LocalDegreeProfile
# from utils import assert_0_based_nids
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from NSUBS.model.OurSGM.data_loader import _get_enc_X
import torch
class NodeFeatureEncoder(object):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        if node_feat_name is None:
            return
        # Go through all the graphs in the entire dataset
        # and create a set of all possible
        # labels so we can one-hot encode them.
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(sorted(inputs_set))}
        self._fit_onehotencoder()

    def _fit_onehotencoder(self):
        self.oe = OneHotEncoder(categories='auto').fit(
            np.array(sorted(self.feat_idx_dic.values())).reshape(-1, 1))

    def encode(self, g):
        assert_0_based_nids(g)  # must be [0, 1, 2, ..., N - 1]
        if self.node_feat_name is None:
            return np.array([[1] for n in sorted(g.nodes())])  # NOTE: this will no longer be called now?
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in sorted(g.nodes())]  # sort nids just to make sure
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)
    
import torch
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std
import networkx as nx

def get_ldf(graph: nx.Graph):
    edge_index = torch.tensor(list(graph.edges())).t().contiguous()
    row, col = edge_index
    N = graph.number_of_nodes()

    deg = torch.tensor([val for (node, val) in sorted(graph.degree(), key=lambda pair: pair[0])], dtype=torch.float)
    deg_col = deg[col]

    min_deg, _ = scatter_min(deg_col, row, dim_size=N)
    min_deg[min_deg > 10000] = 0
    max_deg, _ = scatter_max(deg_col, row, dim_size=N)
    max_deg[max_deg < -10000] = 0
    mean_deg = scatter_mean(deg_col, row, dim_size=N)
    std_deg = scatter_std(deg_col, row, dim_size=N)

    X = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

    return X

def encode_node_features_custom(dataset):
    attr_list = set()
    gs = [g.get_nxgraph() for g in dataset.gs]
    encoder, X = _get_enc_X(gs[0])
    X = encoder.transform(X).toarray()
    # X = torch.tensor(X, dtype=torch.float)
    # gs[0].init_x = X
    for g in gs:
        X = []
        for node, ndata in sorted(g.nodes(data=True)):
            X.append([ndata['type']])
        X = encoder.transform(X).toarray()
        X = torch.tensor(X, dtype=torch.float)
        g.init_x = X
        g.init_x = torch.cat((g.init_x, get_ldf(g)), dim=1)

     
            # self.x = torch.cat((self.x, self.get_ldf()), dim=1)


    return dataset,gs[0].init_x.shape[1]

#    将type属性值放在一个set中
   

     
     
    
    
def node2vec(g):
    embeddings_matrix = np.zeros((len(g.nodes), 64))
    node2vec = Node2Vec(g, dimensions=64, walk_length=30, num_walks=200, workers=1)  
# 训练Node2Vec模型
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  
    # 获取每个节点的嵌入
    for i,node in enumerate(g.nodes):
        embeddings_matrix[i,:] = model.wv[node]
    return embeddings_matrix

def get_onehot(type_val,num_attr):
    onehot = np.zeros(num_attr)  # 创建一个长度为5的零向量，因为有5种类型
    onehot[type_val - 1] = 1  # 将相应的位置设为1
    return onehot

def onehot_attr(g,num_attr):
    nodes = list(g.nodes(data=True))
    onehot_matrix = np.zeros((len(nodes), num_attr))

    for idx, (_, data) in enumerate(nodes):
        type_val = data['type']
        onehot_matrix[idx] = get_onehot(type_val,num_attr)
    return onehot_matrix

def encode_node_features(dataset=None, pyg_single_g=None):
    if dataset:
        assert pyg_single_g is None
        input_dim = 0
    else:
        assert pyg_single_g is not None
        input_dim = pyg_single_g.x.shape[1]
    # node_feat_encoders = get_flags_with_prefix_as_list('node_fe')
    node_feat_encoders = ['one_hot','local_degree_profile']
    if 'one_hot' not in node_feat_encoders:
        raise ValueError('Must have one hot node feature encoder!')
    for nfe in node_feat_encoders:
        if nfe == 'one_hot':
            if dataset:
                input_dim = _one_hot_encode(dataset, input_dim)
        elif nfe == 'local_degree_profile':
            input_dim += 5
            if pyg_single_g:
                pyg_single_g = LocalDegreeProfile()(pyg_single_g)
        else:
            raise ValueError('Unknown node feature encoder {}'.format(nfe))
    if input_dim <= 0:
        raise ValueError('Must have at least one node feature encoder '
                         'so that input_dim > 0')
    if dataset:
        return dataset, input_dim
    else:
        return pyg_single_g, input_dim


def _one_hot_encode(dataset, input_dim):
    gs = [g.get_nxgraph() for g in dataset.gs] # TODO: encode image's complete graph

    from config import FLAGS
    # natts = FLAGS.node_feats_for_sm.split(',')
    natts = FLAGS.node_feats_for_sm
    natts = [] if natts == ['None'] else natts #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # natts = []

    # if len(dataset.natts) > 1:
    if len(natts) > 1:
        node_feat_name = None
        raise ValueError('TODO: handle multiple node features')
    # elif len(dataset.natts) == 1:
    #     node_feat_name = dataset.natts[0]
    elif len(natts) == 1:
        node_feat_name = natts[0]
    else:
        #if no node feat return 1
        for g in gs:
            g.init_x = np.ones((nx.number_of_nodes(g), 1))
        return 1
    nfe = NodeFeatureEncoder(gs, node_feat_name)
    for g in gs:
        x = nfe.encode(g)
        g.init_x = x # assign the initial features
    input_dim += nfe.input_dim()
    return input_dim


def obtain_nfe_feat_idx_div(dataset, natts):
    gs = [g.get_nxgraph() for g in dataset.gs] # TODO: encode image's complete graph
    natts = [] if (natts == ['None'] or natts == 'None') else natts #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # if len(dataset.natts) > 1:
    if len(natts) > 1:
        node_feat_name = None
        raise ValueError('TODO: handle multiple node features')
    # elif len(dataset.natts) == 1:
    #     node_feat_name = dataset.natts[0]
    elif len(natts) == 1:
        node_feat_name = natts[0]
    else:
        return {}

    nfe = NodeFeatureEncoder(gs, node_feat_name)
    return nfe.feat_idx_dic