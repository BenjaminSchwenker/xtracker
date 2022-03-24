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
  <output>upgradeVTXOnlyTrackingValidation.root</output>
  <description>
  This module validates that the vtx only track finding is capable of reconstructing tracks in Y(4S) runs.
  </description>
</header>
"""

from tracking.path_utils import add_hit_preparation_modules
from xtracker.path_utils import add_vtx_track_finding_gnn
from tracking.validation.run import TrackingValidationRun
import logging
import basf2
VALIDATION_OUTPUT_FILE = 'upgradeVTXOnlyTrackingValidation.root'
N_EVENTS = 1000
ACTIVE = True

basf2.set_random_seed(1337)


def setupFinderModule(path):

    import yaml
    import os
    config_path = os.path.expandvars("${XTRACKER_CONFIG_PATH}/belle2_vtx.yaml")

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    add_hit_preparation_modules(path, components=["VTX"], useVTX=True)
    add_vtx_track_finding_gnn(
        path=path,
        reco_tracks="RecoTracks",
        model_path=config['training']['checkpoint'],
        event_cuts=config['event_cuts'],
        segment_cuts=config['selection'],
        tracker_config=config['model'],
    )


class VTXStandalone(TrackingValidationRun):
    """
    Validation class for the N Layer VTX tracking
    """
    #: the number of events to process
    n_events = N_EVENTS
    #: Generator to be used in the simulation (-so)
    generator_module = 'generic'
    #: root input file to use, generated by central validation script
    root_input_file = '../VTXEvtGenSimNoBkg.root'
    #: use full detector for validation
    components = None

    #: lambda method which is used by the validation to add the vtx finder modules
    finder_module = staticmethod(setupFinderModule)

    #: use only the vtx hits when computing efficiencies
    tracking_coverage = {
        'WhichParticles': ['VTX'],  # Include all particles seen in the VTX detector, also secondaries
        'UsePXDHits': False,
        'UseSVDHits': False,
        'UseCDCHits': False,
        'UseVTXHits': True,
    }

    #: perform fit after track finding
    fit_tracks = True
    #: plot pull distributions
    pulls = True
    #: output file of plots
    output_file_name = VALIDATION_OUTPUT_FILE


def main():
    """
    create VTX validation class and execute
    """
    validation_run = VTXStandalone()
    validation_run.configure_and_execute_from_commandline()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if ACTIVE:
        main()
