from read_tve_imsm import read_tve
from sgm_graph import SRW_RWF_ISRW, random_walk_sampling

from utils import create_dir_if_not_exists
from utils_our import print_g

from os.path import dirname, basename, relpath, join
import networkx as nx
from networkx.utils import open_file
from random import Random
from tqdm import tqdm
from glob import glob

QSIZE = 128


def convert_to_veq(from_path, to_path):
    create_dir_if_not_exists(to_path)
    gs_paths = glob(f'{from_path}/**/**/*.graph')
    print(f'Found {len(gs_paths)}')
    # assert len(gs_paths) == 201
    for i, g_path in enumerate(tqdm(sorted(gs_paths))):
        g = read_tve(g_path)
        rp = relpath(g_path, from_path)
        full_path = join(to_path, rp)
        g.gid = i
        create_dir_if_not_exists(dirname(full_path))
        write_veq(g, full_path)


def write_veq(g, to_path):
    with open(to_path, 'w') as fp:
        fp.write(f'#{g.gid}\n')
        fp.write(f'{g.number_of_nodes()}\n')
        for node, ndata in sorted(g.nodes(data=True)):
            fp.write(f'{ndata["label"]}\n')
        fp.write(f'{g.number_of_edges()}\n')
        for e1, e2 in sorted(g.edges()):
            fp.write(f'{e1} {e2}\n')


def main():
    # from_path = '/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset/youtube'
    # to_path = '/home/anonymous/Documents/GraphMatching/model/VEQ/youtube'

    from_path = '/home/anonymous/Documents/GraphMatching/model/SubgraphMatching/dataset'
    to_path = '/home/anonymous/Documents/GraphMatching/model/VEQ/dataset'

    # g = nx.Graph()
    # g.add_edge(0, 1)
    # g.add_edge(1, 0)
    # print_g(g)
    convert_to_veq(from_path, to_path)


if __name__ == '__main__':
    main()
