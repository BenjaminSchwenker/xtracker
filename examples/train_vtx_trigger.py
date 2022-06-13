#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

"""
Performs training of xtracker neural net using prepared training samples
defined in train config file.

Usage:
export BELLE2_VTX_UPGRADE_GT=upgrade_2022-01-21_vtx_5layer
export BELLE2_VTX_BACKGROUND_DIR=/path/to/bgfiles/
export XTRACKER_CONFIG_PATH=<b2>/xtracker/examples/configs/configfile.yaml
export HEP_DATA=<some/path/with/storage>
python3 train_vtx_trigger.py  configs/belle2_vtx_trigger.yaml
"""


import logging
import yaml
import argparse
import os
import pickle

import torch
import numpy as np
import random

from xtracker.gnn_tracking.CoachTrigger import CoachTrigger
from xtracker.gnn_tracking.TrackingGame import TrackingGame as Game
from xtracker.gnn_tracking.pytorch.NNet import NNetWrapperTrigger as nn
from xtracker.datasets import get_data_loaders
from xtracker.utils import dotdict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train_vtx_trigger.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/belle2_vtx_trigger.yaml')
    return parser.parse_args()


def load_config(config_file, **kwargs):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key, val in kwargs.items():
        config[key] = val
    return config


def save_config(config):
    output_dir = config['training']['checkpoint']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)


def setupTracker(game, config_path):
    from xtracker.gnn_tracking.pytorch.NNet import NNetWrapper as tnet
    from xtracker.gnn_tracking.ImTracker import ImTracker
    import yaml
    import os

    config_path = os.path.expandvars(config_path)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tracker_args = dotdict(config['model'])

    # Need to adjust this to checkpoints from training on graphs
    checkpoint_dir = os.path.expandvars(config['training']['checkpoint'])

    # Load neural net
    n1 = tnet(config['networks']['embedding_dim'],config['networks']['tracker_layer_size'],config['networks']['n_update_iters'])
    n1.load_checkpoint(checkpoint_dir, 'best.pth.tar')

    # Built a tracker
    tracker = ImTracker(game, n1, tracker_args)

    return tracker


def main():
    """Main function"""

    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

    # Parse the command line
    cmd_args = parse_args()

    # Load configuration
    config = load_config(cmd_args.config)

    args = dotdict(config)
    args.training = dotdict(args.training)
    args.model = dotdict(args.model)
    args.optimizer = dotdict(args.optimizer)
    args.data = dotdict(args.data)

    # Reproducible training [NOTE, doesn't full work on GPU]
    seed = config['global']['rndseed']
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed + 10)
    random.seed(seed)

    # Data loaders for training and validation
    train_data_loader, valid_data_loader = get_data_loaders(**args.data, input_dir=config['global']['graph_dir'])
    logging.info('Loaded %g training samples', len(train_data_loader.dataset))
    if valid_data_loader is not None:
        logging.info('Loaded %g validation samples', len(valid_data_loader.dataset))

    g = Game(train_data_loader, valid_data_loader)
    logging.info('Loaded %s...', Game.__name__)

    # Setup pretrained tracker
    tracker = setupTracker(g, config_path=config['global']['tracker_config'])

    # Setup untrained trigger
    nnet = nn(config['networks']['embedding_dim'],config['networks']['trigger_layer_size'])
    logging.info('Loaded %s...', nn.__name__)

    if args.training.load_model:
        logging.info('Loading checkpoint "%s/%s"...', args.training.load_folder_file)
        nnet.load_checkpoint(args.training.load_folder_file[0], args.training.load_folder_file[1])
    else:
        logging.warning('Not loading a checkpoint!')

    logging.info('Loading the Coach...')
    c = CoachTrigger(g, tracker, nnet, args)

    if args.training.load_model:
        logging.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    logging.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
