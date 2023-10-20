import networkx as nx
from networkx.utils import open_file

from sklearn.preprocessing import OneHotEncoder
@open_file(0, mode='rb')
def read_tve(path):
    encoding = 'utf-8'
    print(path)
    lines = (line.decode(encoding) for line in path)

    create_using = None
    G = nx.empty_graph(0, create_using)

    comments = '#'
    num_nodes, num_edges = None, None
    degrees = {}
    for line in lines:
        p = line.find(comments)
        if p >= 0:
            line = line[:p]
        if not len(line):
            continue
        # split line, should have 2 or more
        s = line.strip().split(' ')
        if len(s) < 3:
            raise ValueError('Wrong format')
        mode = s.pop(0)
        if mode == 't':
            num_nodes = int(s.pop(0))
            num_edges = int(s.pop(0))
        elif mode == 'v':
            u = int(s.pop(0))
            node_label = int(s.pop(0))
            node_degree = int(s.pop(0))
            degrees[u] = node_degree
            G.add_node(u, label=node_label)
        elif mode == 'e':
            u = int(s.pop(0))
            v = int(s.pop(0))
            if len(s) >= 1:
                assert len(s) == 1
                edge_label = int(s.pop(0))
                if edge_label != 0:
                    raise ValueError(f'Edge label != 0 {line} {edge_label}')
            G.add_edge(u, v)
        else:
            raise ValueError(f'Unrecognized line {line}')

    if G.number_of_nodes() != num_nodes:
        raise ValueError(f'Number of nodes {G.number_of_nodes()} != {num_nodes}')
    if G.number_of_edges() != num_edges:
        raise ValueError(f'Number of edges {G.number_of_edges()} != {num_edges}')
    for node in G.nodes():
        if G.degree[node] != degrees[node]:
            raise ValueError(f'Node {node} has degree {G.degree[node]} '
                             f'but file says {degrees[node]}')

    return G


import networkx as nx
from io import BytesIO
# from data_loader import get_data_loader_imsm_merged

def _create_fit_onehotencoder(gt):
    enc = OneHotEncoder()
    X = []
    X_flat = []
    for node, ndata in sorted(gt.nodes(data=True)):
        X.append([ndata['label']])
        X_flat.append(ndata['label'])
    # print_stats(X_flat, 'node labels', print_func=saver.log_info)
    enc.fit(X)
    # saver.log_info(f'Fit one hot encoder')
    return enc, X

def _get_enc_X(gt):
    enc, X = _create_fit_onehotencoder(gt)
    return enc, X

def main():
    # 1. Mock a file using the expected format.
    mock_file_content = '''t 3 2
v 0 1 2
v 1 2 1
v 2 3 1
e 0 1 0
e 0 2 0
'''
    mock_file = BytesIO(mock_file_content.encode('utf-8'))

    # 2. Call the function to get the graph.
    G = read_tve(mock_file)
    enc, X = _get_enc_X(G)
    gt_X_enc = (G, X, enc)

    # 3. Check if the graph matches our expectation.
    assert G.number_of_nodes() == 3, "Expected 3 nodes"
    assert G.number_of_edges() == 2, "Expected 2 edges"
    assert G.nodes[0]['label'] == 1, "Node 0 should have label 1"
    assert G.nodes[1]['label'] == 2, "Node 1 should have label 2"
    assert G.nodes[2]['label'] == 3, "Node 2 should have label 3"
    assert G.has_edge(0, 1), "Edge between 0 and 1 is missing"
    assert G.has_edge(0, 2), "Edge between 0 and 2 is missing"

    print("All tests passed!")




if __name__ == '__main__':
    main()
