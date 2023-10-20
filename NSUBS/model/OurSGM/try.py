from read_tve_imsm import read_tve
import networkx as nx

g = '/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/youtube_unlabeled/query_graph/query_dense_128_142.graph'
g = read_tve(g)
nx.write_gexf(g, '/home/anonymous/Documents/GraphMatching/model/OurSGM/temp/142.gexf')

# from utils import load_pickle
#
# x = load_pickle('/home/anonymous/Documents/GraphMatching/model/OurSGM/logs/our_test_imsm-yeast_dense_4_GraphQL_1_2022-05-12T00-34-52.730402/obj/CS_cpp.pickle')
# print(x)

# import numpy as np
#
# underlying = np.array([0.07])
# W = 0
# v = underlying
#
# W += v.item()
# print(W, v)
# W += v
# print(W, v)
# W += v
# print(W, v)

#
# import numpy as np
# import networkx as nx
#
# class Thing():
#     pass
#
# thing = Thing()
# thing.x = np.array([0])
#
# G = nx.Graph()
# G.add_node(0, xxx=thing.x)
# print(G.nodes[0])
#
# thing.x += 99999
# print(G.nodes[0])

# def dl():
#     print('here')
#     for i in range(100):
#         print('@@@', i)
#         yield i
#
# d = dl()
# print('@@')
# for d in dl():
#     print('*')
#     print(d)


# import networkx as nx
# import numpy as np
# import random
#
# from copy import deepcopy
#
# get_relabel_map = lambda g: {nid:i for i, nid in enumerate(g.nodes)}
# q_size_li = []
#
# for i in range(50):
#     gt = nx.barabasi_albert_graph(2000, 5)
#     gt = nx.relabel_nodes(gt, get_relabel_map(gt))
#     for nid in gt:
#         gt.nodes[nid]['label'] = 0
#     gq = nx.subgraph(gt, random.sample(gt.nodes, 50))
#     gq = deepcopy(nx.subgraph(gq, max(nx.connected_components(gq), key=len)))
#     assert nx.is_isomorphic(gq, gt.subgraph(gq.nodes))
#     gq = nx.relabel_nodes(gq, get_relabel_map(gq))
#     q_size_li.append(gq.number_of_nodes())
#     nx.write_gexf(gq, f'train/{i}q.gexf')
#     nx.write_gexf(gt, f'train/{i}t.gexf')
#
# for i in range(50):
#     gt = nx.barabasi_albert_graph(2000, 5)
#     gt = nx.relabel_nodes(gt, get_relabel_map(gt))
#     for nid in gt:
#         gt.nodes[nid]['label'] = 0
#     gq = nx.subgraph(gt, random.sample(gt.nodes, 50))
#     gq = deepcopy(nx.subgraph(gq, max(nx.connected_components(gq), key=len)))
#     assert nx.is_isomorphic(gq, gt.subgraph(gq.nodes))
#     gq = nx.relabel_nodes(gq, get_relabel_map(gq))
#     q_size_li.append(gq.number_of_nodes())
#     nx.write_gexf(gq, f'test/{i}q.gexf')
#     nx.write_gexf(gt, f'test/{i}t.gexf')
#
# q_size_li = np.array(q_size_li)
# print(f'stats:\nmean={q_size_li.mean()}\nmax={q_size_li.max()}\nmin={q_size_li.min()}\nstd={q_size_li.std()}')

# zzz = 0


# from utils import load_pickle
# f = '/home/anonymous/Documents/GraphMatching/model/OurSGM/logs/baseline/daf_imsm-eu2005_2021-10-17T01-48-47.480921/smts/test_0_smts_smts.pickle'
# x = load_pickle(f)
# print()

