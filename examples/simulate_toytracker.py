#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################


"""
Script to simulate toy MC for training of track finder.

The simulated detector is a 6 layer pixel vertex detector. The sensitive volumes
are are six thin cylindrical silicon layers. It is a toy simulation neglecting material
and detector resolution effects.

Usage: python3 simulate_toytracker.py  configs/sim_toytracker.yaml
"""

# System
import os
import argparse
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

import xtracker.fastsim.fastsim as toyMC


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('simulate_toytracker.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/configs/sim_toytracker.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('--outputdir', type=str, default='./')
    add_arg('--rndseed', type=int, default=12345)
    return parser.parse_args()


def process_event(
    evtid, num_tracks_min, num_tracks_max, num_noise_min, num_noise_max,
    theta_min, theta_max, p_min, p_max, output_dir,
):
    # Simulate the data
    logging.info('Event %i, generate data' % evtid)

    num_tracks = np.random.randint(num_tracks_min, num_tracks_max)
    num_noise = np.random.randint(num_noise_min, num_noise_max)

    hits, truth, particles = toyMC.make_event(
        num_tracks, num_noise,
        thetaMin=theta_min, thetaMax=theta_max,
        phiMin=0, phiMax=2 * np.pi,
        pMin=p_min, pMax=p_max, charge=None
    )

    filename = output_dir + '/{}_id_{}.h5'.format("graph", evtid)
    hits.to_hdf(filename, key='hits')
    truth.to_hdf(filename, key='truth')
    particles.to_hdf(filename, key='particles')


def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Prepare output
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Writing outputs to ' + output_dir)

    # We generate events on the fly, just need to know how many
    events = range(config['n_events'])

    # Process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir, **config['simulation'])
        pool.map(process_func, events)

    logging.info('All done!')


if __name__ == '__main__':
    main()
