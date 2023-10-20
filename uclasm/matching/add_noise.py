import sys 
sys.path.extend([
        "/home/kli16/ISM_custom/esm_NSUBS/esm",
        # "/home/kli16/ISM_custom/esm_NSUBS/esm/rlmodel",
        "/home/kli16/ISM_custom/esm_NSUBS/esm/uclasm/",
        "/home/kli16/ISM_custom/esm_NSUBS/esm/NSUBS/",
    ])
import networkx as nx
import random
import csv
import networkx as nx
from collections import deque
from graph_pair import GraphPair
from dataset import OurDataset
from graph import RegularGraph
from node_feat import encode_node_features_custom,encode_node_features
import pickle

# Global constants
CSV_FILE_PATH = './tutorial/world_email.csv'
TXT_FILE_PATH = "./tutorial/email-Eu-core-department-labels.txt"
DENSITY_THRESHOLD = 0.2
MAX_NODES_PER_SUBGRAPH = 8
N = 2000
NOISE_EDGES = 5

def add_random_edges(graph, num_edges):
    """
    向图中添加num_edges条随机边
    """
    all_nodes = list(graph.nodes())
    for _ in range(num_edges):
        # 随机选择两个节点
        u, v = random.sample(all_nodes, 2)
        # 确保两个节点之间没有边
        while graph.has_edge(u, v):
            u, v = random.sample(all_nodes, 2)
        graph.add_edge(u, v)



def shuttle_node_id(G):
    nodes = list(G.nodes())
    random.shuffle(nodes)

    # 创建一个映射，将原始节点映射到新的随机节点
    mapping = {original: new for original, new in zip(G.nodes(), nodes)}

    # 使用映射创建一个新的DiGraph
    H = nx.relabel_nodes(G, mapping)
    return H


def rename_id(G):
    nodes_list = list(G.nodes())
    node_id_mapping = {node: idx for idx, node in enumerate(nodes_list)}
    node_id_mapping_reverse = {idx: node for idx, node in enumerate(nodes_list)}
    G = nx.relabel_nodes(G, node_id_mapping, copy=True)
    return G,node_id_mapping_reverse 


def read_csv_edges(file_path):
    edges = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source = int(row['Source'])
            target = int(row['Target'])
            edges.append((source, target))
    return edges


def create_directed_graph(edges):
    # Create a directed graph
    graph = nx.Graph()
    # Add edges from the list to the graph
    graph.add_edges_from(edges)
    return graph


def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into individual elements
            elements = line.strip().split()
            # Convert elements to integers (if applicable)
            elements = [int(e) for e in elements]
            # Append the list of elements to the data list
            data.append(elements)
    return data


def get_subgraph_density(g, nodes):
    """获取给定节点子集的图的密度"""
    subgraph = g.subgraph(nodes)
    return nx.density(subgraph)


def random_dense_subgraph(g, size, density_threshold, n):
    """从图中随机选择n个子图，其中每个子图包含size个节点且密度大于density_threshold"""
    valid_subgraphs = []
    
    while len(valid_subgraphs) < n:
        nodes = random.sample(g.nodes(), size)
        if get_subgraph_density(g, nodes) > density_threshold and nx.is_connected(g.subgraph(nodes)):
            valid_subgraphs.append(nx.Graph(g.subgraph(nodes)))
            print(len(valid_subgraphs))
            
    return valid_subgraphs



def main():

    
    edges_list = read_csv_edges(CSV_FILE_PATH)
    world = create_directed_graph(edges_list)
    txt_data = read_txt_file(TXT_FILE_PATH)
    
    attr_dict = {line[0]: {'type':line[1]} for line in txt_data}
    nx.set_node_attributes(world , attr_dict)

    world.graph['gid'] = 0
    world,_ = rename_id(world)
    graph_list = [RegularGraph(world)]
    pairs = {}
    high_density_subgraphs = random_dense_subgraph(world, MAX_NODES_PER_SUBGRAPH, DENSITY_THRESHOLD, N)
    matchings = []
    for i in range(N):
        sampled_subgraph = high_density_subgraphs[i]
        add_random_edges(sampled_subgraph, NOISE_EDGES)
        sampled_subgraph.graph['gid'] = i + 1
        sampled_subgraph,matching = rename_id(sampled_subgraph)
        matchings.append(matching)
        graph_list.append(RegularGraph(sampled_subgraph))
        pairs[(i+1, 0)] = GraphPair()

    name = 'email'
    natts = ['type']
    eatts = [] 
    tvt = 'train'
    align_metric = 'sm'
    node_ordering = 'bfs'
    glabel = None

    our_dataset = OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering, glabel, None)


    for i in range(len(matchings)):
        g1 = our_dataset.gs[i+1].nxgraph
        g2 = our_dataset.gs[0].nxgraph
        for u,v in matchings[i].items():
            assert(g1.nodes[u]['type'] == g2.nodes[v]['type'])

    dataset_train, _ = encode_node_features_custom(dataset=our_dataset)
    with open(f'./data/unEmail_trainset_dens_{DENSITY_THRESHOLD}_n_{MAX_NODES_PER_SUBGRAPH}_num_{N}_noise_{NOISE_EDGES}_10_18.pkl','wb') as f:
        pickle.dump(dataset_train,f)
    with open(f'./data/unEmail_trainset_dens_{DENSITY_THRESHOLD}_n_{MAX_NODES_PER_SUBGRAPH}_num_{N}_noise_{NOISE_EDGES}_10_18_matching.pkl','wb') as f:
        pickle.dump(matchings,f)


if __name__ == "__main__":
    main()


