# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate


def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'hitgraphs':
        from . import hitgraphs
        train_dataset, valid_dataset = hitgraphs.get_datasets(**data_args)
        collate_fn = hitgraphs.collate_fn
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
    train_sampler, valid_sampler = None, None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
        valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                   shuffle=(train_sampler is None), **loader_args)
    valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader
