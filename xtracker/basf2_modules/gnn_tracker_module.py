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
import ROOT
from ROOT import Belle2
from ROOT import TVector3, TMatrixDSym, TVectorD
import basf2 as b2


from xtracker.gnn_tracking.ImTracker import ImTracker
from xtracker.gnn_tracking.TrackingGame import TrackingGame as Game
from xtracker.gnn_tracking.pytorch.NNet import NNetWrapper as NNet
from xtracker.utils import dotdict as dotdict
from xtracker.track_creation import compute_tracks_from_graph_ml
from xtracker.datasets.graph import Graph, graph_to_sparse, get_batch
from xtracker.graph_creation import make_graph
from xtracker.track_seed_state_retriever import getSeedState


class GNNTracker(b2.Module):
    """Module to form tracks from hits in the tracking detector using graph neural network tracking."""

    def __init__(
        self,
        modelPath,
        n_det_layers,
        cdcHitsColumnName='CDCHits',
        trackCandidatesColumnName="RecoTracks",
    ):
        """Constructor"""

        super(GNNTracker, self).__init__()

        #: cached value of model path
        self.model_path = modelPath
        #: cached name of the CDCHits StoreArray
        self.cdcHitsColumnname = cdcHitsColumnName
        #: cached name of the RecoTracks StoreArray
        self.trackCandidatesColumnName = trackCandidatesColumnName
        #: Min pt in GeV
        self.pt_min = 0.
        #: phi cuts for graph segments
        self.phi_range = (-np.pi, np.pi)
        #: eta cuts for graph segments
        self.eta_range = [-5, 5]
        #: Number of phi sections (subdivision of event)
        self.n_phi_sections = 1
        #: Number of eta sections (subdivision of event)
        self.n_eta_sections = 1
        #: Number of tracking layers
        self.n_det_layers = n_det_layers
        #: Segment type
        self.segment_type = 'all'
        #: Max phi slope
        self.phi_slope_max = 2.1
        #: Max z0
        self.z0_max = 100
        #: Scaling applied to r coordinate of hit (graph creation)
        self.feature_scale_r = 15.0
        #: Scaling applied to phi coordinate of hit (graph creation)
        self.feature_scale_phi = 1.0
        #: Scaling applied to z coordinate of hit (graph creation)
        self.feature_scale_z = 50.0

    def initialize(self):
        """Receive signal at the start of event processing"""

        #: Tracking game object, views tracking as a one player game
        self.game = Game(train_data_loader=None, valid_data_loader=None)

        #: Neural net object to predict hit graph
        self.net = NNet()
        self.net.load_checkpoint(self.model_path, 'best.pth.tar')

        #: Tracker object
        tracker_args = dotdict({
            'nIter': 20,
            'verbose': False,
            'mid_steps': 4,
            'noise_sigma': 0.0,
            'update_e': False,
        })

        self.tracker = ImTracker(self.game, self.net, tracker_args)

        self.tracksStoreArrayHelper = Belle2.TrackStoreArrayHelper(self.trackCandidatesColumnName)

    def event(self):
        """Event method"""

        event_meta_data = Belle2.PyStoreObj("EventMetaData")
        evtid = event_meta_data.getEvent()

        # Fill data structurs for tracking event
        hits, truth, particles, hits_info = self.examine_event()

        # Make initial event graphs
        graphs, IDs = make_graph(
            hits,
            truth,
            particles,
            evtid,
            n_det_layers=self.n_det_layers,
            pt_min=self.pt_min,
            phi_range=self.phi_range,
            n_phi_sections=self.n_phi_sections,
            eta_range=self.eta_range,
            n_eta_sections=self.n_eta_sections,
            segment_type=self.segment_type,
            z0_max=self.z0_max,
            phi_slope_max=self.phi_slope_max,
            feature_scale_r=self.feature_scale_r,
            feature_scale_phi=self.feature_scale_phi,
            feature_scale_z=self.feature_scale_z,
        )

        # Change format, anyoing
        batch = get_batch(graph_to_sparse(graphs[0]))

        # Predict track segments on graphs
        board = self.game.getInitBoardFromBatch(batch)
        preds, score = self.tracker.process(board)

        # Create tracks
        tracks, tracks_qi = compute_tracks_from_graph_ml(board.x, board.edge_index, preds=preds)

        vtxClusters = Belle2.PyStoreArray("VTXClusters")

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

                trackingDetector, arrayIndex = hits_info[hitID]

                if trackingDetector == Belle2.RecoHitInformation.c_VTX:
                    newRecoTrack.addVTXHit(vtxClusters[arrayIndex], hitCounter)
                    hitCounter += 1

    def examine_event(self):

        # cdcHits = Belle2.PyStoreArray(self.cdcHitsColumnname)

        # Running counter used to enumerate all hits in this event
        hit_id = 0

        # Output dictionaries with all relevant event information
        all_hits = {'particle_id': [], 'layer': [], 'x': [], 'y': [], 'z': [], 't': [], 'hit_id': []}
        all_truth = {'hit_id': [], 'particle_id': [], 'weight': []}
        all_particles = {'vx': [], 'vy': [], 'vz': [], 'px': [], 'py': [], 'pz': [], 'q': [], 'nhits': [], 'particle_id': []}
        all_hit_info = []

        # We need to invent mother particle for hits
        nid = -1

        all_particles['particle_id'].append(nid)
        all_particles['px'].append(0)
        all_particles['py'].append(0)
        all_particles['pz'].append(0)
        all_particles['vx'].append(0.0)
        all_particles['vy'].append(0.0)
        all_particles['vz'].append(0.0)
        all_particles['q'].append(0)
        all_particles['nhits'].append(0)

        vtxClusters = Belle2.PyStoreArray("VTXClusters")
        for hit in vtxClusters:

            layer = hit.getSensorID().getLayerNumber()
            sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
            position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

            all_hit_info.append((Belle2.RecoHitInformation.c_VTX, hit.getArrayIndex()))

            all_hits['x'].append(position.X())
            all_hits['y'].append(position.Y())
            all_hits['z'].append(position.Z())
            all_hits['t'].append(0)
            all_hits['layer'].append(layer - 1)
            all_hits['hit_id'].append(hit_id)
            all_hits['particle_id'].append(nid)

            all_truth['hit_id'].append(hit_id)
            all_truth['particle_id'].append(nid)
            all_truth['weight'].append(0)

            hit_id += 1

        return pd.DataFrame(all_hits), pd.DataFrame(all_truth), pd.DataFrame(all_particles), all_hit_info
