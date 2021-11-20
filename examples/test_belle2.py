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
Usage: Set global tag and run
export BELLE2_VTX_UPGRADE_GT=upgrade_2021-07-16_vtx_5layer
basf2 test_belle2.py -n 10
"""

import basf2 as b2
from ROOT import Belle2
import generators as ge
import simulation as si
import reconstruction as re
import mdst as mdst
import glob as glob
import ROOT as r


import os
import sys
from xtracker.path_utils import add_vtx_track_finding_gnn
from vtx import add_vtx_reconstruction, get_upgrade_globaltag

# Need to use default global tag prepended with upgrade GT
b2.conditions.disable_globaltag_replay()
b2.conditions.prepend_globaltag(get_upgrade_globaltag())

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

evtgenInput = b2.register_module('EvtGenInput')
evtgenInput.logging.log_level = b2.LogLevel.WARNING
main.add_module(evtgenInput)

# ---------------------------------------------------------------------------------------

# Detector Simulation:
si.add_simulation(path=main, useVTX=True)

# needed for fitting
main.add_module('SetupGenfitExtrapolation')

# VTX reconstruction
add_vtx_reconstruction(path=main)

# VTX standalone track finding
add_vtx_track_finding_gnn(path=main, components=["VTX"], model_path="tracking/data/gnn_vtx", n_det_layers=5)

# track fitting
daffitter = b2.register_module("DAFRecoFitter")
daffitter.param('recoTracksStoreArrayName', "RecoTracks")
# daffitter.logging.log_level = b2.LogLevel.DEBUG
main.add_module(daffitter)
# also used in the tracking sequence (multi hypothesis)
main.add_module('TrackCreator', recoTrackColName="RecoTracks", pdgCodes=[211, 321, 2212])


# main.add_module("PrintCollections")

b2.print_path(main)

b2.process(main)
print(b2.statistics)
