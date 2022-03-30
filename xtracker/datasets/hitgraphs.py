# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


"""
PyTorch specification for the hit graph dataset.
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def load_graph(filename):
    with np.load(filename) as f:
        x, y, p = f['X'], f['y'], f['P']
        Ri_rows, Ri_cols = f['Ri_rows'], f['Ri_cols']
        Ro_rows, Ro_cols = f['Ro_rows'], f['Ro_cols']
        n_edges = Ri_cols.shape[0]
        edge_index = np.zeros((2, n_edges), dtype=int)
        edge_index[0, Ro_cols] = Ro_rows
        edge_index[1, Ri_cols] = Ri_rows
    return x, edge_index, y, p


class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir, n_samples=None):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.startswith('graph') and not f.endswith('_ID.npz')]
        self.filenames = (
            filenames[:n_samples] if n_samples is not None else filenames)

    def __getitem__(self, index):
        return load_graph(self.filenames[index])

    def __len__(self):
        return len(self.filenames)


def get_datasets(input_dir, n_train, n_valid):
    data = HitGraphDataset(input_dir, n_train + n_valid)
    # deterministic splitting ensures all workers split the same way
    torch.manual_seed(1)
    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    return train_data, valid_data


def collate_fn(graphs):
    """
    Collate function for building mini-batches from a list of hit-graphs.
    This function should be passed to the pytorch DataLoader.
    It will stack the hit graph matrices sized according to the maximum
    sizes in the batch and padded with zeros.

    This implementation could probably be optimized further.
    """
    batch_size = len(graphs)

    # Special handling of batch size 1
    if batch_size == 1:
        g = graphs[0]
        x, edge_index, y, p = g
        return [torch.from_numpy(m).float() for m in g]

    else:
        print('Not supported ')
        sys.exit(1)
