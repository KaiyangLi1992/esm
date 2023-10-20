from NSUBS.model.OurSGM.read_tve_imsm import read_tve
from NSUBS.model.OurSGM.sgm_graph import SRW_RWF_ISRW, random_walk_sampling

from NSUBS.src.utils import create_dir_if_not_exists
from NSUBS.model.OurSGM.utils_our import print_g

from os.path import dirname, relpath, join, basename
import networkx as nx
from networkx.utils import open_file
from random import Random
from tqdm import tqdm
from glob import glob



def convert_snap_and_sample(from_path, to_path_target, to_path_query):
    create_dir_if_not_exists(dirname(to_path_target))
    create_dir_if_not_exists(dirname(to_path_query))
    gt = nx.Graph()
    with open(from_path) as fp:
        cnt = 0
        line = fp.readline()
        while line:
            cnt += 1
            if line[0] != '#':
                ls = line.split()
                assert len(ls) == 2
                from_id, to_id = int(ls[0]), int(ls[1])
                gt.add_edge(from_id, to_id)
            line = fp.readline()

    gt = nx.convert_node_labels_to_integers(gt)

    print(f'{cnt} lines')
    print_g(gt, 'loaded g')

    write_tve(gt, to_path_target)

    # print('done; check now')
    # g = read_tve(to_path)
    # print_g(g, 'loaded tvt g')

def sample_q(gt, to_path_query, qsizes):

    srw_rwf_isrw = SRW_RWF_ISRW(Random(345))
    for qsize in qsizes:
        print('qsize', qsize)
        for qsize in tqdm([qsize]):
            for i in tqdm(range(200)):
                gq = random_walk_sampling(gt, qsize, srw_rwf_isrw, check_connected=False)
                # node labels/colors are copied/sampled from the original target graph
                gq = nx.convert_node_labels_to_integers(gq)
                fn = f'{to_path_query}_{qsize}_{i+1}.graph'
                print(fn)
                write_tve(gq, fn, unlabeled=False)


def write_tve(g, to_path, unlabeled=True):
    with open(to_path, 'w') as fp:
        fp.write(f't {g.number_of_nodes()} {g.number_of_edges()}\n')
        for node, ndata in sorted(g.nodes(data=True)):
            if unlabeled:
                label = 0
            else:
                label = ndata.get('label', 0)
            fp.write(f'v {node} {label} {g.degree[node]}\n')
        for e1, e2 in sorted(g.edges()):
            fp.write(f'e {e1} {e2}\n')



def convert_tve_unlabeled(from_folder, to_folder, target_name):
    assert from_folder != to_folder
    paths = glob(f'{from_folder}/**/*.graph', recursive=True)
    print(f'{len(paths)} files to convert')
    for path in tqdm(sorted(paths)):
        g = read_tve(path)
        rp = relpath(path, from_folder)
        if 'data_graph' in path:
            rp = rp.replace(f'{target_name}.graph', f'{target_name}_unlabeled.graph')
        full_path = join(to_folder, rp)
        create_dir_if_not_exists(dirname(full_path))
        # print(full_path)
        write_tve(g, full_path)






def main():
    # from_path = '/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/roadNet-CA.txt'
    # to_path_target = '/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/roadNet-CA/data_graph/roadNet-CA.graph'
    # to_path_query = '/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/roadNet-CA/query_graph/query_dense'


    # g = nx.Graph()
    # g.add_edge(0, 1)
    # g.add_edge(1, 0)
    # print_g(g)
    # convert_snap_and_sample(from_path, to_path_target, to_path_query)

    # target_name = 'patents'
    # target_name = 'youtube'
    # target_name = 'eu2005'
    # target_name = 'roadNet-CA'
    # target_name = 'yeast'
    # target_name = 'dblp'
    # target_name = 'wordnet'
    # target_name = 'hprd'
    target_name = 'human'



    tp = f'/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/{target_name}/data_graph/{target_name}.graph'
    to_path_query = f'/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/{target_name}/query_graph/query_dense'
    gt = read_tve(tp)
    sample_q(gt, to_path_query, [64, 128])


    from_folder = f'/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/{target_name}/'
    to_folder = f'/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/{target_name}_unlabeled'
    convert_tve_unlabeled(from_folder, to_folder, target_name)


if __name__ == '__main__':
    main()
