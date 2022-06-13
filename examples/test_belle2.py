#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

"""
Simple script to generate a sample of Belle II events.
Output fie can be displayed later with b2display tool.

Usage: Set global tag and run
export BELLE2_VTX_UPGRADE_GT=upgrade_2022-01-21_vtx_5layer
basf2 test_belle2.py -n 10 -- configs/belle2_vtx.yaml
"""

import basf2 as b2
from ROOT import Belle2
import generators as ge
import simulation as si
import reconstruction as re
import mdst as mdst
import glob as glob
import ROOT as r

import argparse
import os
import sys
import yaml
from xtracker.path_utils import add_vtx_track_finding_gnn, add_track_printer
from vtx import add_vtx_reconstruction, get_upgrade_globaltag


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('test_belle2.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/belle2_vtx.yaml')
    add_arg("--gen", default='bbar', help="Generator: 'gun', 'bbar'")
    return parser.parse_args()


def main():
    """Main function"""

    # Need to use default global tag prepended with upgrade GT
    b2.conditions.disable_globaltag_replay()
    b2.conditions.prepend_globaltag(get_upgrade_globaltag())

    # Parse the command line
    args = parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create basf2 path
    path = b2.create_path()

    eventinfosetter = b2.register_module('EventInfoSetter')
    # default phase3 geometry:
    exp_number = 0
    eventinfosetter.param("expList", [exp_number])
    path.add_module(eventinfosetter)

    eventinfoprinter = b2.register_module('EventInfoPrinter')
    path.add_module(eventinfoprinter)

    progress = b2.register_module('Progress')
    path.add_module(progress)

    # simulation settings
    if args.gen == 'bbar':
        evtgenInput = b2.register_module('EvtGenInput')
        evtgenInput.logging.log_level = b2.LogLevel.WARNING
        path.add_module(evtgenInput)
    elif args.gen == 'gun':
        path.add_module(
            'ParticleGun',
            pdgCodes=[-211, 211],
            momentumParams=[0.05, 3.0],
            xVertexParams=[0.0],
            yVertexParams=[0.0],
            zVertexParams=[0.0]
        )

    # Detector Simulation:
    si.add_simulation(path=path, useVTX=True)

    # needed for fitting
    path.add_module('SetupGenfitExtrapolation')

    # VTX reconstruction
    add_vtx_reconstruction(path=path)

    # Setting up the neural network based track finder
    add_vtx_track_finding_gnn(
        path=path,
        reco_tracks="RecoTracks",
        model_path=config['training']['checkpoint'],
        event_cuts=config['event_cuts'],
        segment_cuts=config['selection'],
        tracker_config=config['model'],
    )

    # track fitting
    daffitter = b2.register_module("DAFRecoFitter")
    daffitter.param('recoTracksStoreArrayName', "RecoTracks")
    # daffitter.logging.log_level = b2.LogLevel.DEBUG
    path.add_module(daffitter)
    # also used in the tracking sequence (multi hypothesis)
    path.add_module('TrackCreator', recoTrackColName="RecoTracks", pdgCodes=[211, 321, 2212])

    path.add_module("PrintCollections")

    # add output
    path.add_module('RootOutput')

    b2.print_path(path)

    b2.process(path)
    print(b2.statistics)


if __name__ == '__main__':
    main()
