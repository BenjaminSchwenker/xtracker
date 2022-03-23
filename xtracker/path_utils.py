##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

from basf2 import register_module

from xtracker.basf2_modules.gnn_tracker_module import GNNTracker
from xtracker.basf2_modules.track_printer_module import TrackPrinter


def add_vtx_track_finding_gnn(
    path,
    vtx_clusters="",
    reco_tracks="RecoTracks",
    components=None,
    suffix="",
    model_path="tracking/data/gnn_vtx",
    event_cuts={},
    segment_cuts={},
    tracker_config={},
):
    """
    Convenience function for adding all graph neural network track finder modules
    to the path. The function considers VTX and CDC hits for tracking.

    The result is a StoreArray with name @param reco_tracks full of RecoTracks.

    :param path: basf2 path
    :param vtx_clusters: VTXCluster collection name
    :param reco_tracks: Name of the output RecoTracks, Defaults to RecoTracks.
    :param components: List of the detector components to be used in the reconstruction. Defaults to None which means
                       all components.
    :param suffix: all names of intermediate Storearrays will have the suffix appended. Useful in cases someone needs to
                   put several instances of track finding in one path.
    :param model_path: if set to a finite value, neural network state will be loaded from file instead of the database.
    :param event_cuts: dictionary of parameters for selecting event data for tracking
    :param segment_cuts: dictionary of parameters for selecting segments between hits
    :param tracker_config: dictionary of additional tracker configurations
    """

    # setup the event level tracking info to log errors and stuff
    nameTrackingInfoModule = "RegisterEventLevelTrackingInfo" + suffix
    nameEventTrackingInfo = "EventLevelTrackingInfo" + suffix
    if nameTrackingInfoModule not in path:
        # use modified name of module and created StoreObj
        registerEventlevelTrackingInfo = register_module('RegisterEventLevelTrackingInfo')
        registerEventlevelTrackingInfo.set_name(nameTrackingInfoModule)
        registerEventlevelTrackingInfo.param('EventLevelTrackingInfoName', nameEventTrackingInfo)
        path.add_module(registerEventlevelTrackingInfo)

    # add the tracker
    tracker = GNNTracker(
        model_path=model_path,
        event_cuts=event_cuts,
        segment_cuts=segment_cuts,
        tracker_config=tracker_config,
        trackCandidatesColumnName=reco_tracks)
    path.add_module(tracker)


def add_track_printer(
    path,
    reco_tracks="RecoTracks",
    mc_reco_tracks="MCRecoTracks",
    printSimHits=False,
):
    printer = TrackPrinter(
        trackCandidatesColumnName=reco_tracks,
        mcTrackCandidatesColumName=mc_reco_tracks,
        printSimHits=printSimHits,
    )
    path.add_module(printer)
