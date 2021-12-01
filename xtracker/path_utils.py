##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

from pybasf2 import B2WARNING
from ROOT import Belle2
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
    n_det_layers=5,
):
    """
    Convenience function for adding all graph neural network vxd track finder modules
    to the path. The function only considers VTX hits for VTX standalone
    tracking.

    The result is a StoreArray with name @param reco_tracks full of RecoTracks (not TrackCands any more!).
    Use the GenfitTrackCandidatesCreator Module to convert back.

    :param path: basf2 path
    :param vtx_clusters: VTXCluster collection name
    :param reco_tracks: Name of the output RecoTracks, Defaults to RecoTracks.
    :param components: List of the detector components to be used in the reconstruction. Defaults to None which means
                       all components.
    :param suffix: all names of intermediate Storearrays will have the suffix appended. Useful in cases someone needs to
                   put several instances of track finding in one path.
    :param model_path: if set to a finite value, neural network state will be loaded from file instead of the database.
    :param n_det_layers: Number of vertex detector layers used for tracking
    """

    # setup the event level tracking info to log errors and stuff
    nameTrackingInfoModule = "RegisterEventLevelTrackingInfo" + suffix
    nameEventTrackingInfo = "EventLevelTrackingInfo" + suffix
    if nameTrackingInfoModule not in path:
        # Use modified name of module and created StoreObj
        registerEventlevelTrackingInfo = register_module('RegisterEventLevelTrackingInfo')
        registerEventlevelTrackingInfo.set_name(nameTrackingInfoModule)
        registerEventlevelTrackingInfo.param('EventLevelTrackingInfoName', nameEventTrackingInfo)
        path.add_module(registerEventlevelTrackingInfo)

    # add the tracker
    tracker = GNNTracker(
        modelPath=Belle2.FileSystem.findFile(model_path),
        n_det_layers=n_det_layers,
        trackCandidatesColumnName=reco_tracks)
    path.add_module(tracker)


def add_track_printer(
    path,
    reco_tracks="RecoTracks"
):
    printer = TrackPrinter(trackCandidatesColumnName=reco_tracks)
    path.add_module(printer)
