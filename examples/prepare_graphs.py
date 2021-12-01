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
Data preparation script for training of xtracker.

Prepares hitgraphs from simulated events for training. Can be used for
Belle II MC and simplified toytracker.

Usage:
python3 prepare_graphs.py configs/prep_graphs_belle2.yaml

or

python3 prepare_graphs.py configs/prep_graphs_toytracker.yaml
"""

import os
import argparse
import logging
from pathlib import Path
import multiprocessing as mp
from functools import partial

import yaml
import numpy as np
import pandas as pd

from xtracker.datasets.graph import Graph, save_graphs
from xtracker.graph_creation import (
    calc_dphi, calc_eta, select_segments, construct_graph, select_hits, split_detector_sections, form_layer_pairs
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare_graphs.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prep_graphs_belle2.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    return parser.parse_args()


def process_event(
    evtid, output_dir, pt_min, n_eta_sections, n_phi_sections,
    eta_range, phi_range, phi_slope_max, z0_max, input_dir, segment_type, n_det_layers,
    feature_scale_r, feature_scale_phi, feature_scale_z
):

    # Read the data
    logging.info('Event %i, read event' % evtid)

    if not Path(input_dir + '/graph_id_{}.h5'.format(evtid)).is_file():
        logging.info('Event %i not found.' % evtid)
        return

    hits = pd.read_hdf(os.path.expandvars(input_dir + '/graph_id_{}.h5'.format(evtid)), 'hits')
    truth = pd.read_hdf(os.path.expandvars(input_dir + '/graph_id_{}.h5'.format(evtid)), 'truth')
    particles = pd.read_hdf(os.path.expandvars(input_dir + '/graph_id_{}.h5'.format(evtid)), 'particles')

    # Read the data
    logging.info('Event %i, generate graph' % evtid)

    # Apply hit selection
    logging.info('Event %i, selecting hits' % evtid)
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(evtid=evtid)

    # Divide detector into sections
    phi_edges = np.linspace(*phi_range, num=n_phi_sections + 1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections + 1)
    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)

    # Construct layer pairs
    layer_pairs = form_layer_pairs(n_det_layers, segment_type)

    # Graph features and scale
    feature_names = ['r', 'phi', 'z']

    # Scale hit coordinates
    feature_scale = np.array([feature_scale_r, np.pi / n_phi_sections / feature_scale_phi, feature_scale_z])

    # Construct the graph
    logging.info('Event %i, constructing graphs' % evtid)
    graphs_all = [construct_graph(section_hits, layer_pairs=layer_pairs,
                                  phi_slope_max=phi_slope_max, z0_max=z0_max,
                                  feature_names=feature_names,
                                  feature_scale=feature_scale)
                  for section_hits in hits_sections]
    graphs = [x[0] for x in graphs_all]
    IDs = [x[1] for x in graphs_all]

    # Write these graphs to the output directory
    try:
        filenames = [os.path.join(output_dir, 'event_%04i_g%03i' % (evtid, i))
                     for i in range(len(graphs))]
        filenames_ID = [os.path.join(output_dir, 'event_%04i_g%03i_ID' % (evtid, i))
                        for i in range(len(graphs))]
    except Exception as e:
        logging.info(e)
    logging.info('Event %i, writing graphs', evtid)
    save_graphs(graphs, filenames)
    for ID, file_name in zip(IDs, filenames_ID):
        np.savez(file_name, ID=ID)


def main():
    """Main function"""

    # Parse the command line
    args = parse_args()

    # Setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Prepare output
    input_dir = os.path.expandvars(config['input_dir'])
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    logging.info('Writing outputs to ' + output_dir)

    # We generate events on the fly, just need to know how many
    events = range(config['n_events'])

    # Process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir, input_dir=input_dir,
                               phi_range=(-np.pi, np.pi), **config['selection'])
        pool.map(process_func, events)

    logging.info('All done!')


if __name__ == '__main__':
    main()
