#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


"""
Script to simulate Belle II MC for training of vtx track trigger.

The simulated detector uses the upgraded vertex detector with a fully pixelated
vertex detector (VTX) and the current Central Drift Chamber (CDC). The exact detector
geometry is set by the environment variable BELLE2_VTX_UPGRADE_GT.

Usage:
export BELLE2_VTX_UPGRADE_GT=upgrade_2022-01-21_vtx_5layer
basf2 simulate_vtx_trigger.py -- configs/belle2_vtx.yaml
"""

import argparse
import os
import yaml

import basf2 as b2
from simulation import add_simulation
from xtracker.basf2_modules.event_collector_module import TrackingEventCollector
from vtx import get_upgrade_globaltag
from tracking.path_utils import add_hit_preparation_modules


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('simulate_vtx_trigger.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/belle2_vtx.yaml')
    add_arg('--bbbar', action='store_true', help='Simulate BBBar events as signal')
    return parser.parse_args()


def add_collector_path(path, config, output_dir, isSignalEvent):
    """Adds all modules to path up to event collector module for sampling training events"""

    add_simulation(path=path, useVTX=True, simulateT0jitter=False,)

    add_hit_preparation_modules(
        path=path,
        components=None,
        useVTX=True,
        useVTXClusterShapes=True
    )

    path.add_module(
        "TFCDC_WireHitPreparer",
        wirePosition="aligned",
        useSecondHits=False,
        flightTimeEstimation="outwards",
        filter="cuts_from_DB"
    )

    mctrackfinder = path.add_module('TrackFinderMCTruthRecoTracks')
    for param, value in config['mctrackfinder'].items():
        mctrackfinder.param(param, value)

    # Building tracking events for training xtracker
    event_collector = TrackingEventCollector(
        output_dir_name=output_dir,
        event_cuts=config['event_cuts'],
        isSignalEvent=isSignalEvent
    )
    path.add_module(event_collector)


def main():
    """Main function"""

    # Use default global tag prepended with upgrade GT to replace PXD+SVD by VTX
    b2.conditions.disable_globaltag_replay()
    b2.conditions.prepend_globaltag(get_upgrade_globaltag())

    # Parse the command line
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Prepare output
    output_dir = os.path.expandvars(config['global']['event_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # Number of events to simulate j
    n_events = config['global']['n_events']

    # Set Random Seed for reproducable simulation.
    rndseed = config['global']['rndseed']
    b2.set_random_seed(rndseed)

    # Set log level. Can be overridden with the "-l LEVEL" flag for basf2.
    b2.set_log_level(b2.LogLevel.WARNING)

    # ---------------------------------------------------------------------------------------
    path = b2.create_path()

    eventinfosetter = path.add_module('EventInfoSetter')
    eventinfosetter.param("expList", [0])
    if args.bbbar:
        eventinfosetter.param("evtNumList", [int(n_events/2)])
        eventinfosetter.param("skipNEvents", 0)
    else:
        eventinfosetter.param("evtNumList", [n_events])
        eventinfosetter.param("skipNEvents", int(n_events/2))

    path.add_module('EventInfoPrinter')
    path.add_module('Progress')

    if args.bbbar:
        evtgenInput = path.add_module('EvtGenInput')
        evtgenInput.logging.log_level = b2.LogLevel.WARNING
        add_collector_path(path, config, output_dir=output_dir, isSignalEvent=1.0)
    else:
        add_collector_path(path, config, output_dir=output_dir, isSignalEvent=0.0)

    b2.print_path(path)

    b2.process(path)
    print(b2.statistics)


if __name__ == '__main__':
    main()
