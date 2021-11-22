##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

import math
import ROOT
from ROOT import Belle2
from ROOT import TVector3
import basf2 as b2

from tracking.validation.utilities import (
    getHelixFromMCParticle,
    getSeedTrackFitResult,
    is_primary,
    get_det_hit_ids,
    calc_ndf_from_det_hit_ids,
    getObjectList,
)

import numpy as np
import pandas as pd
import collections

import os


class TrackingEventCollector(b2.Module):
    """Module to collect tracking information per event and write it to an output directory. The set of
    tracking events will later be used to train tracking algorithms."""

    def __init__(
        self,
        name="VTXDefault",
        output_dir_name=".",
        cdcHitsColumnName='CDCHits',
        trackCandidatesColumnName="RecoTracks",
        mcTrackCandidatesColumName="MCRecoTracks",
    ):
        """Constructor"""

        super(TrackingEventCollector, self).__init__()

        #: cached value of the tracking-collector name
        self.collector_name = name
        #: cached value of the output dir
        self.output_dir_name = output_dir_name
        #: cached name of the CDCHits StoreArray
        self.cdcHitsColumnname = cdcHitsColumnName
        #: cached name of the RecoTracks StoreArray
        self.trackCandidatesColumnName = trackCandidatesColumnName
        #: cached name of the MCRecoTracks StoreArray
        self.mcTrackCandidatesColumnName = mcTrackCandidatesColumName

    def initialize(self):
        """Receive signal at the start of event processing"""

        #: Track-match object that examines relation information from MCMatcherTracksModule
        self.trackMatchLookUp = Belle2.TrackMatchLookUp(self.mcTrackCandidatesColumnName, self.trackCandidatesColumnName)

    def event(self):
        """Event method"""

        event_meta_data = Belle2.PyStoreObj("EventMetaData")
        evtid = event_meta_data.getEvent() - 1  # first evtid is zero

        event_hits, event_truth, event_particles = self.examine_event()

        filename = self.output_dir_name + '/{}_id_{}.h5'.format("graph", evtid)
        event_hits.to_hdf(filename, key='hits')
        event_truth.to_hdf(filename, key='truth')
        event_particles.to_hdf(filename, key='particles')

    def examine_event(self):

        # Initialze variables to be returned
        all_hits = {'particle_id': [], 'layer': [], 'x': [], 'y': [], 'z': [], 't': [], 'hit_id': []}
        all_truth = {'hit_id': [], 'particle_id': [], 'weight': []}
        all_particles = {'vx': [], 'vy': [], 'vz': [], 'px': [], 'py': [], 'pz': [], 'q': [], 'nhits': [], 'particle_id': []}

        trackMatchLookUp = self.trackMatchLookUp

        # Analyse from the Monte Carlo reference side
        mcTrackCands = Belle2.PyStoreArray(self.mcTrackCandidatesColumnName)
        mcParticles = Belle2.PyStoreArray('MCParticles')
        if not mcTrackCands:
            return pd.DataFrame(all_hits), pd.DataFrame(all_truth), pd.DataFrame(all_particles)

        cdcHits = Belle2.PyStoreArray(self.cdcHitsColumnname)

        used_vtx_hits = []
        used_pxd_hits = []
        used_cdc_hits = []

        # Running counter used to enumerate all hits in this event
        hit_id = 0

        for i, mcTrackCand in enumerate(mcTrackCands):

            mcParticle = trackMatchLookUp.getRelatedMCParticle(mcTrackCand)
            mcHelix = getHelixFromMCParticle(mcParticle)

            momentum = mcParticle.getMomentum()
            vertex = mcParticle.getVertex()
            charge = mcParticle.getCharge()

            pt = momentum.Perp()
            tan_lambda = np.divide(1.0, math.tan(momentum.Theta()))  # Avoid zero division exception
            d0 = mcHelix.getD0()
            det_hit_ids = get_det_hit_ids(mcTrackCand)
            ndf = calc_ndf_from_det_hit_ids(det_hit_ids)
            nhits = len(det_hit_ids)

            all_particles['particle_id'].append(i)
            all_particles['px'].append(momentum.X())
            all_particles['py'].append(momentum.Y())
            all_particles['pz'].append(momentum.Z())
            all_particles['vx'].append(vertex.X())
            all_particles['vy'].append(vertex.Y())
            all_particles['vz'].append(vertex.Z())
            all_particles['q'].append(charge)
            all_particles['nhits'].append(nhits)

            # Loop over all hits
            for hit_info in mcTrackCand.getRelationsWith("RecoHitInformations"):
                layer = np.float("nan")
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_SVD:
                    hit = hit_info.getRelated("SVDClusters")
                    layer = hit.getSensorID().getLayerNumber()
                    used_svd_hits.append(hit.getArrayIndex())
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_PXD:
                    hit = hit_info.getRelated("PXDClusters")
                    layer = hit.getSensorID().getLayerNumber()
                    used_pxd_hits.append(hit.getArrayIndex())
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_VTX:
                    hit = hit_info.getRelated("VTXClusters")
                    layer = hit.getSensorID().getLayerNumber()
                    used_vtx_hits.append(hit.getArrayIndex())

                    sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
                    position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

                    all_hits['x'].append(position.X())
                    all_hits['y'].append(position.Y())
                    all_hits['z'].append(position.Z())
                    all_hits['t'].append(0.0)
                    all_hits['layer'].append(layer - 1)   # xtracker starts layer numbering internally at 0 not 1 as in Belle II
                    all_hits['hit_id'].append(hit_id)
                    all_hits['particle_id'].append(i)

                    all_truth['hit_id'].append(hit_id)
                    all_truth['particle_id'].append(i)
                    all_truth['weight'].append(0)

                    hit_id += 1

                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_CDC:
                    hit = hit_info.getRelated("CDCHits")
                    layer = hit.getISuperLayer()
                    used_svd_hits.append(hit.getArrayIndex())

        # We need to invent mother particle for noise hits
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

            if hit.getArrayIndex() in used_vtx_hits:
                continue

            layer = hit.getSensorID().getLayerNumber()
            sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
            position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

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

        return pd.DataFrame(all_hits), pd.DataFrame(all_truth), pd.DataFrame(all_particles)
