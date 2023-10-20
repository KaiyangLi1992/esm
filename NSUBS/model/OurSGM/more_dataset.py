import os
import os.path as osp

import torch
import numpy as np
from torch_sparse import coalesce
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_gz, extract_tar)
from torch_geometric.data.makedirs import makedirs


def read_graph(files, name, tissue):
    import pandas as pd

    skiprows = 4
    # if name == 'pokec':
    #     skiprows = 0

    df = pd.read_csv(files[0], sep='\t', header=None, skiprows=1)
    # print(df.shape)
    # from collections import Counter
    # from pprint import pprint
    # pprint(Counter(df[2]).most_common())
    df_new = df.loc[df[2] == tissue]
    # print(df_new.shape)
    # exit()
    edge_index = df_new[[0, 1]]

    # edge_index = pd.read_csv(files[0], sep='\t', header=None,
    #                          skiprows=1, dtype=np.int64, usecols=[0,2])
    num_nodes = len(np.unique(edge_index.values))
    edge_index = torch.from_numpy(edge_index.values).t()
    # num_nodes = edge_index.max().item() + 1
    # edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


class BioSNAPDataset(InMemoryDataset):
    url = 'https://snap.stanford.edu/biodata/datasets'

    available_datasets = {
        'ppi-ohmnet_tissues': ['10013/files/PPT-Ohmnet_tissues-combined.edgelist.gz'],
    }

    def __init__(self, root, name, tissue, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name.lower()
        self.tissue = tissue
        assert self.name in self.available_datasets.keys(), ''
        super(BioSNAPDataset, self).__init__(root, transform, pre_transform,
                                             pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for name in self.available_datasets[self.name]:
            path = download_url('{}/{}'.format(self.url, name), self.raw_dir)
            if name.endswith('.tar.gz'):
                extract_tar(path, self.raw_dir)
            elif name.endswith('.gz'):
                extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        raw_dir = self.raw_dir
        filenames = os.listdir(self.raw_dir)
        if len(filenames) == 1 and osp.isdir(osp.join(raw_dir, filenames[0])):
            raw_dir = osp.join(raw_dir, filenames[0])

        raw_files = sorted([osp.join(raw_dir, f) for f in os.listdir(raw_dir)])

        data_list = read_graph(raw_files, self.name[4:], self.tissue)

        if len(data_list) > 1 and self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return 'BioSNAP-{}({})'.format(self.name, len(self))
