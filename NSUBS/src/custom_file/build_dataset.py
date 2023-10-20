import networkx as nx
import pickle

def load_dblp_snap_file(filename):
    G = nx.Graph()
    
    with open(filename, 'r') as f:
        for line in f:
            # 忽略注释和其他非边的行
            if line.startswith("#"):
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v)
    
    return G
import networkx as nx

# 读取txt文件并获取所有的子图节点
def read_txt_file(filename):
    with open(filename, 'r') as file:
        return [list(map(int, line.strip().split('\t'))) for line in file]

# 从主图中提取子图
def extract_subgraphs(main_graph, nodes_list, min_nodes, max_nodes):
    subgraphs = []
    for nodes in nodes_list:
        if min_nodes <= len(nodes) <= max_nodes:
            subgraph = main_graph.subgraph(nodes).copy()
            subgraphs.append(subgraph)
    return subgraphs

# 主函数
# if __name__ == "__main__":
    # 假设你的nxgraph文件名是'nxgraph.gml'
    # main_graph = nx.read_gml('nxgraph.gml')

    # nodes_list = read_txt_file('your_txt_file.txt')

    # small_subgraphs = extract_subgraphs(main_graph, nodes_list, 20, 30)
    # large_subgraphs = extract_subgraphs(main_graph, nodes_list, 30, 60)

    # 确保只获取50个子图
    # small_subgraphs = small_subgraphs[:50]
    # large_subgraphs = large_subgraphs[:50]

    print(f"Found {len(small_subgraphs)} small subgraphs and {len(large_subgraphs)} large subgraphs.")


# 使用函数
# filename = "/home/kli16/NSUBS/src/custom_file/raw_data/com-dblp.ungraph.txt"
# dblp_graph = load_dblp_snap_file(filename)
# with open('/home/kli16/NSUBS/model/SubgraphMatching/dataset/dblp_unlabeled/data_graph/target.pickle','wb') as f:
#     pickle.dump(dblp_graph,f)

# nodes_list = read_txt_file('/home/kli16/NSUBS/src/custom_file/raw_data/com-dblp.all.cmty.txt')
# small_subgraphs = extract_subgraphs(dblp_graph, nodes_list, 20, 30)
# large_subgraphs = extract_subgraphs(dblp_graph, nodes_list, 30, 60)
# small_subgraphs = small_subgraphs[:50]
# large_subgraphs = large_subgraphs[:50]

# for i in range(0,50):
#     with open(f'/home/kli16/NSUBS/model/SubgraphMatching/dataset/dblp_unlabeled/query_graph/query_dense_64_{i}.graph','wb') as f:
#         pickle.dump(small_subgraphs[i],f)

# for i in range(50,100):
#     with open(f'/home/kli16/NSUBS/model/SubgraphMatching/dataset/dblp_unlabeled/query_graph/query_dense_64_{i}.graph','wb') as f:
#         pickle.dump(large_subgraphs[i-50],f)

import random

def generate_graph_data(filename, num_nodes=100, num_edges=300):
    # 首先创建节点数据
    nodes_data = []
    for i in range(num_nodes):
        label = random.randint(1, 5)  # 假设节点标签是1到5之间的随机整数
        degree = 0  # 初始度数设置为0，稍后会根据边数据进行更新
        nodes_data.append((i, label, degree))

    # 创建边数据
    edges_data = []
    while len(edges_data) < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in edges_data and (v, u) not in edges_data:  # 确保没有自环或重复的边
            edges_data.append((u, v))

    # 调整节点数据以反映实际的度数
    for u, v in edges_data:
        nodes_data[u] = (nodes_data[u][0], nodes_data[u][1], nodes_data[u][2] + 1)
        nodes_data[v] = (nodes_data[v][0], nodes_data[v][1], nodes_data[v][2] + 1)

    # 将数据写入文件
    with open(filename, 'w') as f:
        f.write(f"t {num_nodes} {num_edges}\n")
        for node in nodes_data:
            f.write(f"v {node[0]} {node[1]} {node[2]}\n")
        for edge in edges_data:
            f.write(f"e {edge[0]} {edge[1]}\n")

generate_graph_data("/home/kli16/NSUBS/model/SubgraphMatching/dataset/dblp_unlabeled/data_graph/dblp.graph")



import networkx as nx
import random

def load_graph_from_txt(filename):
    G = nx.Graph()
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts[0] == 'v':
                G.add_node(int(parts[1]), label=int(parts[2]))
            elif parts[0] == 'e':
                G.add_edge(int(parts[1]), int(parts[2]))
    return G

def sample_subgraph(G, size):
    start_node = random.choice(list(G.nodes()))
    visited = set()
    to_visit = [start_node]
    while len(visited) < size and to_visit:
        current_node = to_visit.pop()
        if current_node not in visited:
            visited.add(current_node)
            neighbors = list(G[current_node].keys())
            random.shuffle(neighbors)
            to_visit.extend(neighbors)
    return G.subgraph(visited)

# def write_subgraphs_to_txt(G, output_filename,  size=10):
#     with open(output_filename, 'w') as f:
#         subgraph = sample_subgraph(G, size)
#         f.write(f"t {subgraph.number_of_nodes()} {subgraph.number_of_edges()}\n")
#         for node, data in subgraph.nodes(data=True):
#             f.write(f"v {node} {data['label']} {subgraph.degree(node)}\n")
#         for u, v, data in subgraph.edges(data=True):
#             f.write(f"e {u} {v} {data['label']}\n")

def write_subgraphs_to_txt(G, output_filename, size=10):
    with open(output_filename, 'w') as f:
        subgraph = sample_subgraph(G, size)
        
        # Create a mapping from the old node IDs to the new node IDs
        node_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(subgraph.nodes(data=True), start=0)}
        
        f.write(f"t {subgraph.number_of_nodes()} {subgraph.number_of_edges()}\n")
        
        # Use the new node IDs when writing to the file
        for old_id, data in subgraph.nodes(data=True):
            new_id = node_mapping[old_id]
            f.write(f"v {new_id} {data['label']} {subgraph.degree(old_id)}\n")
            
        # Use the new node IDs for edges when writing to the file
        for u, v, data in subgraph.edges(data=True):
            new_u = node_mapping[u]
            new_v = node_mapping[v]
            f.write(f"e {new_u} {new_v}\n")


G = load_graph_from_txt("/home/kli16/NSUBS/model/SubgraphMatching/dataset/dblp_unlabeled/data_graph/dblp.graph")
for i in range(1,201):    
    write_subgraphs_to_txt(G, f"/home/kli16/NSUBS/model/SubgraphMatching/dataset/dblp_unlabeled/query_graph/query_dense_64_{i}.graph")








