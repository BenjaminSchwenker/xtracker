##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

import numpy as np
import pandas as pd
import math
from ROOT import Belle2
from ROOT import TVector3, TMatrixDSym, TVectorD
from pybasf2 import B2WARNING
import basf2 as b2


from xtracker.gnn_tracking.ImTracker import ImTracker
from xtracker.gnn_tracking.TrackingGame import TrackingGame as Game
from xtracker.gnn_tracking.pytorch.NNet import NNetWrapper as NNet
from xtracker.utils import dotdict as dotdict
from xtracker.track_creation import compute_tracks_from_graph_ml
from xtracker.datasets.graph import Graph, graph_to_sparse, get_batch
from xtracker.graph_creation import make_graph
from xtracker.track_seed_state_retriever import getSeedState
from xtracker.event_creation import make_event, make_event_with_mc
from xtracker.gnn_tracking.TrackingSolver import TrackingSolver


class GNNTracker(b2.Module):
    """Module to find tracks from hits in the tracking detector using graph neural network tracking."""

    def __init__(
        self,
        model_path,
        event_cuts,
        segment_cuts,
        tracker_config,
        cdcHitsColumnName='CDCHits',
        trackCandidatesColumnName="RecoTracks",
    ):
        """Constructor"""

        super(GNNTracker, self).__init__()

        #: cached value of model path
        self.model_path = model_path
        #: cached dictionary of cuts defining event data
        self.event_cuts = event_cuts
        #: cached dictionary of cuts for segment selection
        self.segment_cuts = segment_cuts
        #: cached dictionary of params for trackfinding
        self.tracker_config = tracker_config
        #: cached name of the CDCHits StoreArray
        self.cdcHitsColumnname = cdcHitsColumnName
        #: cached name of the RecoTracks StoreArray
        self.trackCandidatesColumnName = trackCandidatesColumnName

    def initialize(self):
        """Receive signal at the start of event processing"""

        #: Tracking game object, views tracking as a one player game
        self.game = Game(train_data_loader=None, valid_data_loader=None)

        #: Tracker object
        tracker_args = dotdict(self.tracker_config)
        if not tracker_args.useMC:
            # Using Imitation tracker
            net = NNet()
            net.load_checkpoint(self.model_path, 'best.pth.tar')
            self.tracker = ImTracker(self.game, net, tracker_args)
        else:
            # Using MC tracking aka TrackingSolver for debugging
            B2WARNING("Using TrackingSolver with MC truth information for debugging.")
            self.tracker = TrackingSolver(self.game)

        #: Helper object for adding RecoTracks to basf2 datastore
        self.tracksStoreArrayHelper = Belle2.TrackStoreArrayHelper(self.trackCandidatesColumnName)

    def event(self):
        """Event method"""

        event_meta_data = Belle2.PyStoreObj("EventMetaData")
        evtid = event_meta_data.getEvent()

        cdcHits = Belle2.PyStoreArray(self.cdcHitsColumnname)
        vtxClusters = Belle2.PyStoreArray("VTXClusters")

        # Create event data structure
        if not self.segment_cuts['useMC']:
            hits, truth, particles, detector_info = make_event(cdcHits, vtxClusters, self.event_cuts)
        else:
            mcTrackCands = Belle2.PyStoreArray("MCRecoTracksHelpMe")
            trackMatchLookUp = Belle2.TrackMatchLookUp("MCRecoTracksHelpMe", "RecoTracks")
            hits, truth, particles, detector_info = make_event_with_mc(
                cdcHits, vtxClusters, trackMatchLookUp, mcTrackCands, self.event_cuts)

        # Make initial hit graphs over event data
        graphs, IDs = make_graph(
            hits,
            truth,
            particles,
            evtid,
            **self.segment_cuts,
            phi_range=(-np.pi, np.pi),
        )

        # Change format, should be streamlined a bit
        batch = get_batch(graph_to_sparse(graphs[0]))

        # Predict track segments on graphs
        board = self.game.getInitBoardFromBatch(batch)
        preds, score = self.tracker.process(board)

        # Create tracks
        tracks, tracks_qi = compute_tracks_from_graph_ml(board.x, board.edge_index, preds=preds)

        for track, qi in zip(tracks, tracks_qi):

            # Select tensor with hits from track candidate
            track_hits = hits[['x', 'y', 'z', 'layer']].iloc[track]

            # track_hits = track_hits.drop_duplicates(subset=['layer'])
            # if len(track_hits) < 3:
            #    continue

            # Hits on first three layers
            x = track_hits[['x', 'y', 'z']].head(3).values

            # Extrapolate the position and momentum to the point of the first hit on the helix
            # with uncertainties
            stateSeed, covSeed, charge = getSeedState(x)
            position = TVector3(stateSeed[0], stateSeed[1], stateSeed[2])
            momentum = TVector3(stateSeed[3], stateSeed[4], stateSeed[5])
            time = 0

            newRecoTrack = self.tracksStoreArrayHelper.addTrack(position, momentum, charge)
            newRecoTrack.setTimeSeed(time)
            newRecoTrack.setSeedCovariance(covSeed)
            newRecoTrack.setQualityIndicator(qi)

            hitCounter = 0

            for hitID in track:

                trackingDetector, arrayIndex = detector_info[hitID]

                if trackingDetector == Belle2.RecoHitInformation.c_VTX:
                    newRecoTrack.addVTXHit(vtxClusters[arrayIndex], hitCounter)
                    hitCounter += 1

                if trackingDetector == Belle2.RecoHitInformation.c_CDC:
                    newRecoTrack.addCDCHit(cdcHits[arrayIndex], hitCounter, Belle2.RecoHitInformation.RightLeftInformation.c_right)
                    hitCounter += 1

                    # TODO: this is how the MCTrackFinder resolves the left/right ambiguity. Need to do something here as well
                    # now determine the correct sign to resolve the left right ambiguity in the fitter
                    # TVector3 simHitPos = aCDCSimHitPtr->getPosTrack();
                    # TVector3 simMom = aCDCSimHitPtr->getMomentum();
                    # TVector3 simHitPosOnWire = aCDCSimHitPtr->getPosWire();

                    # CDC::CDCGeometryPar& cdcGeometry = CDC::CDCGeometryPar::Instance();
                    # const unsigned short isRightHit = cdcGeometry.getNewLeftRightRaw(simHitPosOnWire, simHitPos, simMom);
