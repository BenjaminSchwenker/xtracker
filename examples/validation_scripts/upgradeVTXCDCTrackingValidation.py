#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# basf2 (Belle II Analysis Software Framework)                           #
# Author: The Belle II Collaboration                                     #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

"""
<header>
  <contact>software-tracking@belle2.org</contact>
  <input>VTXEvtGenSimNoBkg.root</input>
  <output>upgradeVTXCDCTrackingValidation.root</output>
  <description>
  This module validates that the vtx+cdc track finding is capable of reconstructing tracks in Y(4S) runs.
  </description>
</header>
"""

from tracking.path_utils import add_hit_preparation_modules
from xtracker.path_utils import add_vtx_track_finding_gnn, add_track_printer
from tracking.validation.run import TrackingValidationRun
import logging
import basf2
VALIDATION_OUTPUT_FILE = 'upgradeVTXCDCTrackingValidation.root'
N_EVENTS = 1000
ACTIVE = True


basf2.set_random_seed(1337)


def setupFinderModule(path):

    import yaml
    import os
    config_path = os.path.expandvars("${XTRACKER_CONFIG_PATH}")

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    add_hit_preparation_modules(path, useVTX=True)

    # Setting up the MC based track finder.
    mctrackfinder = basf2.register_module('TrackFinderMCTruthRecoTracks')
    mctrackfinder.param('RecoTracksStoreArrayName', 'MCRecoTracksHelpMe')
    mctrackfinder.param('UseReassignedHits', True)
    mctrackfinder.param('UseNLoops', 1)
    mctrackfinder.param('WhichParticles', ['primary'])  # [])
    mctrackfinder.param('UseCDCHits', True)
    mctrackfinder.param('UseVTXHits', True)
    mctrackfinder.param('UsePXDHits', True)
    mctrackfinder.param('UseSVDHits', True)
    mctrackfinder.param('UseOnlyBeforeTOP', True)
    path.add_module(mctrackfinder).set_name('MCFinderHelper')

    # Setting up the neural network based track finder
    add_vtx_track_finding_gnn(
        path=path,
        reco_tracks="RecoTracks",
        model_path=config['training']['checkpoint'],
        event_cuts=config['event_cuts'],
        segment_cuts=config['selection'],
        tracker_config=config['model'],
        networks_config=config['networks'],
    )

    # add_track_printer(path, mc_reco_tracks="MCRecoTracksHelpMe", printSimHits=True)


class VTXCDC(TrackingValidationRun):
    """
    Validation class for the N Layer VTX tracking
    """
    #: the number of events to process
    n_events = N_EVENTS
    #: Generator to be used in the simulation (-so)
    generator_module = 'generic'
    #: root input file to use, generated by central validation script
    root_input_file = '../VTXEvtGenSimNoBkg.root'
    #: lambda method which is used by the validation to add the vtx finder modules
    finder_module = staticmethod(setupFinderModule)

    #: use only the vtx cdc hits when computing efficiencies
    tracking_coverage = {
        'WhichParticles': ['primary'],  # []
        'UsePXDHits': False,
        'UseSVDHits': False,
        'UseCDCHits': True,
        'UseVTXHits': True,
        'UseReassignedHits': True,
        'UseNLoops': 1
    }
    #: tracks will be already fitted by
    #: add_tracking_reconstruction finder module set above
    fit_tracks = False
    #: But we need to tell the validation module to use the fit information
    use_fit_information = True
    #: Switch to use the extended harvesting validation instead
    extended = True
    #: Only works in extended mode
    saveFullTrees = True
    #: do not create expert-level output histograms
    use_expert_folder = True
    #: Include pulls in the validation output
    pulls = True
    #: Include resolution information in the validation output
    resolution = True
    #: name of the output ROOT file
    output_file_name = VALIDATION_OUTPUT_FILE


def main():
    """
    create VTX validation class and execute
    """
    validation_run = VTXCDC()
    validation_run.configure_and_execute_from_commandline()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if ACTIVE:
        main()
