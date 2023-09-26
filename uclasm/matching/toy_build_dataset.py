import sys 
sys.path.append("/home/kli16/ISM_custom/esm") 
sys.path.append("/home/kli16/ISM_custom/esm/rlmodel") 
sys.path.append("/home/kli16/ISM_custom/esm/uclasm/") 
print(sys.path)
import uclasm
import networkx as nx
import random
import csv
import networkx as nx
from collections import deque
from graph_pair import GraphPair
from dataset import OurDataset
from graph import RegularGraph
from node_feat import encode_node_features_custom,encode_node_features
from data_model import OurModelData, OurCocktailData
import pickle
import itertools 

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
    G = nx.relabel_nodes(G, node_id_mapping, copy=True)
    return G




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
    graph = nx.DiGraph()
    # Add edges from the list to the graph
    graph.add_edges_from(edges)
    return graph



# Now you can use 'directed_graph' as a NetworkX DiGraph
# print("Nodes:", world.nodes())
# print("Edges:", world.edges())

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
# Replace 'edges.csv' with the actual path to your CSV file
csv_file_path = '../../tutorial/world_email.csv'
edges_list = read_csv_edges(csv_file_path)
world = create_directed_graph(edges_list)
# Replace 'your_file_path.txt' with the actual path to your txt file
file_path = "../../tutorial/email-Eu-core-department-labels.txt"
txt_data = read_txt_file(file_path)
attr_dict ={}
for line in txt_data:
    attr_dict[line[0]] = {'type':line[1]}

nx.set_node_attributes(world , attr_dict)




def get_subgraph_density(g, nodes):
    """获取给定节点子集的图的密度"""
    subgraph = g.subgraph(nodes)
    return nx.density(subgraph)

def random_dense_subgraph(g, size, density_threshold, n):
    """从图中随机选择n个子图，其中每个子图包含size个节点且密度大于density_threshold"""
    valid_subgraphs = []
    
    while len(valid_subgraphs) < n:
        nodes = random.sample(g.nodes(), size)
        if get_subgraph_density(g, nodes) > density_threshold and nx.is_weakly_connected(g.subgraph(nodes)):
            valid_subgraphs.append(nx.DiGraph(g.subgraph(nodes)))
            print(len(valid_subgraphs))
            
    return valid_subgraphs





def bfs_sample_subgraph(graph, start_node, max_depth):
    sampled_graph = nx.DiGraph()
    queue = deque([(start_node, 0)])  # 存储节点和对应的深度，初始深度为0

    while queue:
        node, depth = queue.popleft()

        if depth > max_depth:
            break

        if node not in sampled_graph:
            sampled_graph.add_node(node,type = graph.nodes[node]['type'])
            

        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if neighbor not in sampled_graph:
                # sampled_graph.add_node(neighbor)
                sampled_graph.add_node(neighbor,type = graph.nodes[neighbor]['type'])
                sampled_graph.add_edge(node, neighbor)
                queue.append((neighbor, depth + 1))

    return sampled_graph

world.graph['gid'] = 0

graph_list = []
# high_density_subgraphs = random_dense_subgraph(world, 7, 0.3,1)
# world = high_density_subgraphs[0]
# world = world
world = rename_id(world)
graph_list.append(RegularGraph(world))
# graph_list.append(RegularGraph(rename_id(world)))
pairs = {}

density_threshold = 0.2 # 设定密度阈值
max_nodes_per_subgraph = 8  # 子图的最大点数
n = 50
# density_threshold = 0.3  # 设定密度阈值
# max_nodes_per_subgraph = 8  # 子图的最大点数

high_density_subgraphs = random_dense_subgraph(world, max_nodes_per_subgraph,density_threshold,n)

for i in range(0,n):
    # random_seed = random.choice(list(world.nodes()))
    # sampled_subgraph = bfs_sample_subgraph(world, start_node=random_seed, max_depth=1)
    sampled_subgraph = high_density_subgraphs[i]
    sampled_subgraph.graph['gid'] = i+1
    sampled_subgraph = rename_id(shuttle_node_id(sampled_subgraph))
    graph_list.append(RegularGraph(sampled_subgraph))
    pairs[(i+1, 0)] = GraphPair()

name = 'email'
natts = ['type']
eatts = [] 
tvt = 'train'
align_metric = 'sm'
node_ordering = 'bfs'
glabel = None

our_dataset = OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

dataset_train, num_node_feat_test = \
            encode_node_features_custom(dataset=our_dataset)
# dataset_train = our_dataset
# num_node_feat_test = 104

with open('Email_testset_dens_0.5_n_10.pkl','wb') as f:
    pickle.dump(dataset_train,f)

# dataset_train = OurModelData(dataset_train, num_node_feat_test)
# dataset_trains = [dataset_train]
# toy_dataset = OurCocktailData(dataset_trains,[100])
# with open('toy_dataset.pkl','wb') as f:
#     pickle.dump(toy_dataset,f)

# print(num_node_feat_test)
