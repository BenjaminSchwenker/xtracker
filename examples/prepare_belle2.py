#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

# Usage: basf2 prepare_belle2.py  -n 20000 -- --rndseed=12345 --outputdir=./data/events_belle2_vtxonly


import argparse
import os
import sys
import random
from pathlib import Path

import basf2 as b2
import ROOT as r
from beamparameters import add_beamparameters
from simulation import add_simulation
from xtracker.event_collector_module import TrackingEventCollector
from vtx import add_vtx_reconstruction, get_upgrade_globaltag

# Need to use default global tag prepended with upgrade GT
b2.conditions.disable_globaltag_replay()
b2.conditions.prepend_globaltag(get_upgrade_globaltag())

# ---------------------------------------------------------------------------------------

"""Parse command line arguments."""
parser = argparse.ArgumentParser('prepare_belle2.py')
add_arg = parser.add_argument
add_arg('--outputdir', type=str, default='./')
add_arg('--rndseed', type=int, default=12345)
args = parser.parse_args()


# Set Random Seed for reproducable simulation. 0 means really random.
rndseed = args.rndseed
b2.set_random_seed(rndseed)

# Store all event data here, ok if directory already exists
outputDir = args.outputdir
Path(outputDir).mkdir(parents=True, exist_ok=True)


# Set log level. Can be overridden with the "-l LEVEL" flag for basf2.
b2.set_log_level(b2.LogLevel.WARNING)

# ---------------------------------------------------------------------------------------
main = b2.create_path()

eventinfosetter = b2.register_module('EventInfoSetter')
# default phase3 geometry:
exp_number = 0
eventinfosetter.param("expList", [exp_number])
main.add_module(eventinfosetter)

eventinfoprinter = b2.register_module('EventInfoPrinter')
main.add_module(eventinfoprinter)

progress = b2.register_module('Progress')
main.add_module(progress)

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

print("WARNING: setting non-default beam vertex at x= " + str(vertex_x) + " y= " + str(vertex_y) + " z= " + str(vertex_z))

# Particle Gun:
# One can add more particle gun modules if wanted.

# additional flatly smear the muon vertex between +/- this value
vertex_delta = 0.005  # in cm

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
main.add_module(particlegun)

# EvtGen Simulation:
# TODO: There are newer convenience functions for this task -> Include them!
# Beam parameters
beamparameters = add_beamparameters(main, "Y4S")
beamparameters.param("vertex", [vertex_x, vertex_y, vertex_z])
# print_params(beamparameters)

evtgenInput = b2.register_module('EvtGenInput')
evtgenInput.logging.log_level = b2.LogLevel.WARNING
main.add_module(evtgenInput)


# ---------------------------------------------------------------------------------------

# Detector Simulation:
add_simulation(path=main,
               useVTX=True,
               )

# needed for fitting
main.add_module('SetupGenfitExtrapolation')

add_vtx_reconstruction(path=main)

# Setting up the MC based track finder.
mctrackfinder = b2.register_module('TrackFinderMCTruthRecoTracks')
mctrackfinder.param('UseCDCHits', False)
mctrackfinder.param('UseSVDHits', False)
mctrackfinder.param('UsePXDHits', False)
mctrackfinder.param('UseVTXHits', True)
mctrackfinder.param('Smearing', False)
mctrackfinder.param('MinimalNDF', 6)
mctrackfinder.param('WhichParticles', ['primary'])
mctrackfinder.param('RecoTracksStoreArrayName', 'MCRecoTracks')
# set up the track finder to only use the first half loop of the track and discard all other hits
mctrackfinder.param('UseNLoops', 0.5)
mctrackfinder.param('discardAuxiliaryHits', True)
# mctrackfinder.logging.log_level = b2.LogLevel.DEBUG
# mctrackfinder.logging.debug_level = 2000
main.add_module(mctrackfinder)


# include a track fit into the chain (sequence adopted from the tracking scripts)
# Correct time seed: Do I need it for VXD only tracks ????
main.add_module("IPTrackTimeEstimator", recoTracksStoreArrayName="MCRecoTracks", useFittedInformation=False)

# track fitting
daffitter = b2.register_module("DAFRecoFitter")
daffitter.param('recoTracksStoreArrayName', "MCRecoTracks")
# daffitter.logging.log_level = b2.LogLevel.DEBUG
main.add_module(daffitter)
# also used in the tracking sequence (multi hypothesis)
# may be overkill
# main.add_module('TrackCreator', recoTrackColName="MCRecoTracks", pdgCodes=[211, 321, 2212])

event_collector = TrackingEventCollector(output_dir_name=outputDir)
main.add_module(event_collector)

b2.log_to_file('createSim.log', append=False)

b2.print_path(main)

b2.process(main)
print(b2.statistics)
