import sys 
sys.path.append("/root/uclasm/") 
sys.path.append("/root/uclasm/rlmodel") 
sys.path.append("/root/uclasm/uclasm/") 
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
from node_feat import encode_node_features
from data_model import OurModelData, OurCocktailData
import pickle
import itertools 

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

# Replace 'edges.csv' with the actual path to your CSV file
csv_file_path = './tutorial/world_email.csv'
edges_list = read_csv_edges(csv_file_path)
world = create_directed_graph(edges_list)

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

# Replace 'your_file_path.txt' with the actual path to your txt file
file_path = "./tutorial/email-Eu-core-department-labels.txt"
txt_data = read_txt_file(file_path)
attr_dict ={}
for line in txt_data:
    attr_dict[line[0]] = {'type':line[1]}

nx.set_node_attributes(world , attr_dict)

def find_high_density_subgraphs(g, density_threshold, max_nodes_per_subgraph):
    high_density_subgraphs = []

    # for nodes in g.nodes():
    #     for k in range(2, max_nodes_per_subgraph + 1):
    for subset_nodes in itertools.combinations(g.nodes(), max_nodes_per_subgraph):
        subgraph = g.subgraph(subset_nodes)
        density = nx.density(subgraph)
        if density >= density_threshold:
            high_density_subgraphs.append(subgraph)
        if len(high_density_subgraphs) >= 100:
            break

    return high_density_subgraphs





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
graph_list.append(RegularGraph(rename_id(world)))
pairs = {}

density_threshold = 0.5  # 设定密度阈值
max_nodes_per_subgraph = 10  # 子图的最大点数

high_density_subgraphs = find_high_density_subgraphs(world, density_threshold, max_nodes_per_subgraph)

for i in range(100):
    random_seed = random.choice(list(world.nodes()))
    # sampled_subgraph = bfs_sample_subgraph(world, start_node=random_seed, max_depth=1)
    sampled_subgraph = high_density_subgraphs[i]
    sampled_subgraph.graph['gid'] = i+1
    graph_list.append(RegularGraph(rename_id(sampled_subgraph)))
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
            encode_node_features(dataset=our_dataset)

dataset_train = OurModelData(dataset_train, num_node_feat_test )
dataset_trains = [dataset_train]
toy_dataset = OurCocktailData(dataset_trains,[num_node_feat_test])

with open('toy_dataset.pkl','wb') as f:
    pickle.dump(toy_dataset,f)

print(num_node_feat_test)
