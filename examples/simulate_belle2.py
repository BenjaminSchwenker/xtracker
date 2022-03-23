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
Script to simulate Belle II MC for training of track finder.

The simulated detector uses the upgraded vertex detector with a fully pixelated
vertex detector (VTX) and the current Central Drift Chamber (CDC). The exact detector
geometry is set by the environment variable BELLE2_VTX_UPGRADE_GT.

Usage:
export BELLE2_VTX_UPGRADE_GT=upgrade_2021-07-16_vtx_5layer
basf2 simulate_belle2.py -- configs/belle2_vtx_cdc.yaml
"""

import argparse
import os
import sys
import random
import yaml

import basf2 as b2
import ROOT as r
from beamparameters import add_beamparameters
from simulation import add_simulation
from xtracker.basf2_modules.event_collector_module import TrackingEventCollector
from vtx import add_vtx_reconstruction, get_upgrade_globaltag

# Many scripts import these functions from `tracking`, so leave these imports here
from tracking.path_utils import add_hit_preparation_modules


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('simulate_belle2.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/belle2_vtx_cdc.yaml')
    return parser.parse_args()


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

    eventinfosetter = b2.register_module('EventInfoSetter')
    # default phase3
    exp_number = 0
    eventinfosetter.param("expList", [exp_number])
    eventinfosetter.param("evtNumList", [n_events])
    path.add_module(eventinfosetter)

    eventinfoprinter = b2.register_module('EventInfoPrinter')
    path.add_module(eventinfoprinter)

    progress = b2.register_module('Progress')
    path.add_module(progress)

    # ---------------------------------------------------------------------------------------
    # Simulation Settings:

    # randomize the vertex position (flatly distributed) to make the sectormap more robust wrt. changing beam position
    # minima and maxima of the beam position given in cm
    random.seed(rndseed)
    vertex_x_min = -0.1
    vertex_x_max = 0.1
    vertex_y_min = -0.1
    vertex_y_max = 0.1
    vertex_z_min = -0.5
    vertex_z_max = 0.5

    vertex_x = random.uniform(vertex_x_min, vertex_x_max)
    vertex_y = random.uniform(vertex_y_min, vertex_y_max)
    vertex_z = random.uniform(vertex_z_min, vertex_z_max)
    # additional flatly smear the muon vertex between +/- this value
    vertex_delta = 0.005  # in cm

    print("WARNING: setting non-default beam vertex at x= " + str(vertex_x) + " y= " + str(vertex_y) + " z= " + str(vertex_z))

    # Particle Gun:
    # One can add more particle gun modules if wanted.
    particlegun = b2.register_module('ParticleGun')
    particlegun.logging.log_level = b2.LogLevel.WARNING
    param_pGun = {
        'pdgCodes': [13, -13],   # 13 = muon --> negatively charged!
        'nTracks': 10,
        'momentumGeneration': 'uniform',
        'momentumParams': [0.1, 4],
        'vertexGeneration': 'uniform',
        'xVertexParams': [vertex_x - vertex_delta, vertex_x + vertex_delta],            # in cm...
        'yVertexParams': [vertex_y - vertex_delta, vertex_y + vertex_delta],
        'zVertexParams': [vertex_z - vertex_delta, vertex_z + vertex_delta]
    }

    particlegun.param(param_pGun)
    path.add_module(particlegun)

    # Particle gun for low pt pions
    particlegun_2 = b2.register_module('ParticleGun')
    particlegun_2.set_name('ParticleGun2')
    particlegun_2.logging.log_level = b2.LogLevel.WARNING
    param_pGun_2 = {
        'pdgCodes': [211, -211],   # 211 = pion --> negatively charged!
        'nTracks': 1,
        'momentumGeneration': 'uniform',
        'momentumParams': [0.1, 0.30],
        'vertexGeneration': 'uniform',
        'xVertexParams': [vertex_x - vertex_delta, vertex_x + vertex_delta],            # in cm...
        'yVertexParams': [vertex_y - vertex_delta, vertex_y + vertex_delta],
        'zVertexParams': [vertex_z - vertex_delta, vertex_z + vertex_delta]
    }

    particlegun_2.param(param_pGun_2)
    path.add_module(particlegun_2)

    # EvtGen Simulation:
    # Beam parameters
    beamparameters = add_beamparameters(path, "Y4S")
    beamparameters.param("vertex", [vertex_x, vertex_y, vertex_z])

    evtgenInput = b2.register_module('EvtGenInput')
    evtgenInput.logging.log_level = b2.LogLevel.WARNING
    path.add_module(evtgenInput)

    # ---------------------------------------------------------------------------------------

    # Detector Simulation:
    add_simulation(path=path, useVTX=True,)

    # Needed for track fitting
    path.add_module('SetupGenfitExtrapolation', energyLossBrems=False, noiseBrems=False)

    # Prepare hits for tracking
    add_hit_preparation_modules(
        path=path,
        components=None,
        useVTX=True,
        useVTXClusterShapes=True
    )

    # Parts of CDC TF that may turn out to be usefull here
    # Init the geometry for cdc tracking and the hits and cut low ADC hits
    # path.add_module("TFCDC_WireHitPreparer",
    #                wirePosition="aligned",
    #                useSecondHits=False,
    #                flightTimeEstimation="outwards",
    #                filter="cuts_from_DB")

    # Constructs clusters
    # path.add_module("TFCDC_ClusterPreparer",
    #                ClusterFilter="all",
    #                ClusterFilterParameters={})

    # Find segments within the clusters
    # path.add_module("TFCDC_SegmentFinderFacetAutomaton")

    ####

    # Setting up the MC based track finder.
    mctrackfinder = b2.register_module('TrackFinderMCTruthRecoTracks')
    for param, value in config['simulation']['mctrackfinder'].items():
        mctrackfinder.param(param, value)
    path.add_module(mctrackfinder)

    # Include a track fit into the chain (sequence adopted from the tracking scripts)
    path.add_module("IPTrackTimeEstimator", recoTracksStoreArrayName="MCRecoTracks", useFittedInformation=False)

    # Track fitting
    daffitter = b2.register_module("DAFRecoFitter")
    daffitter.param('recoTracksStoreArrayName', "MCRecoTracks")
    path.add_module(daffitter)

    # Building tracking events for training xtracker
    event_collector = TrackingEventCollector(output_dir_name=output_dir, event_cuts=config['simulation']['event_collector'])
    path.add_module(event_collector)

    b2.print_path(path)

    b2.process(path)
    print(b2.statistics)


if __name__ == '__main__':
    main()
