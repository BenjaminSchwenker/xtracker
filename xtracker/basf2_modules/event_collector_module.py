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
    get_det_hit_ids,
    calc_ndf_from_det_hit_ids,
)

import numpy as np
import pandas as pd
import collections

from xtracker.event_creation import make_empty_event


class TrackingEventCollector(b2.Module):
    """Module to collect tracking information per event and write it to an output directory. The set of
    tracking events will later be used to train tracking algorithms."""

    def __init__(
        self,
        output_dir_name=".",
        cdcHitsColumnName='CDCHits',
        trackCandidatesColumnName="RecoTracks",
        mcTrackCandidatesColumName="MCRecoTracks",
        UseCDCHits=True,
        UseSVDHits=False,
        UsePXDHits=False,
        UseVTXHits=True,
        OffsetVTX=0,
        OffsetPXD=0,
        OffsetSVD=3,
        OffsetCDC=5,
    ):
        """Constructor"""

        super(TrackingEventCollector, self).__init__()

        #: cached value of the output dir
        self.output_dir_name = output_dir_name
        #: cached name of the CDCHits StoreArray
        self.cdcHitsColumnname = cdcHitsColumnName
        #: cached name of the RecoTracks StoreArray
        self.trackCandidatesColumnName = trackCandidatesColumnName
        #: cached name of the MCRecoTracks StoreArray
        self.mcTrackCandidatesColumnName = mcTrackCandidatesColumName
        #: cached flag to use CDC hits
        self.UseCDCHits = UseCDCHits 
        #: cached flag to use SVD hits
        self.UseSVDHits = UseSVDHits
        #: cached flag to use PXD hits
        self.UsePXDHits = UsePXDHits 
        #: cached flag to use VTX hits
        self.UseVTXHits = UseVTXHits 
        #: cached layer offset for VTX
        self.OffsetVTX = OffsetVTX 
        #: cached layer offset for PXD
        self.OffsetPXD = OffsetPXD
        #: cached layer offset for SVD
        self.OffsetSVD = OffsetSVD
        #: cached layer offset for CDC
        self.OffsetCDC = OffsetCDC

    def initialize(self):
        """Receive signal at the start of event processing"""

        #: Track-match object that examines relation information from MCMatcherTracksModule
        self.trackMatchLookUp = Belle2.TrackMatchLookUp(self.mcTrackCandidatesColumnName, self.trackCandidatesColumnName)

        #: Geo-helper object that provides interface to locate CDC wires in space 
        self.cdcGeoHelper = Belle2.CDCGeometryHelper()

    def event(self):
        """Event method"""

        event_meta_data = Belle2.PyStoreObj("EventMetaData")
        evtid = event_meta_data.getEvent() - 1  # first evtid is zero

        hits, truth, particles = self.examine_event()

        filename = self.output_dir_name + '/{}_id_{}.h5'.format("event", evtid)
        hits.to_hdf(filename, key='hits')
        truth.to_hdf(filename, key='truth')
        particles.to_hdf(filename, key='particles')

    def examine_event(self):

        # Create an empty event structure 
        hits, truth, particles, hit_info = make_empty_event()

        trackMatchLookUp = self.trackMatchLookUp

        # Analyse from the Monte Carlo reference side
        mcTrackCands = Belle2.PyStoreArray(self.mcTrackCandidatesColumnName)
        mcParticles = Belle2.PyStoreArray('MCParticles')
        if not mcTrackCands:
            return pd.DataFrame(hits), pd.DataFrame(truth), pd.DataFrame(particles)

        cdcHits = Belle2.PyStoreArray(self.cdcHitsColumnname)
        vtxClusters = Belle2.PyStoreArray("VTXClusters")

        used_vtx_hits = []
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

            particles['particle_id'].append(i)
            particles['px'].append(momentum.X())
            particles['py'].append(momentum.Y())
            particles['pz'].append(momentum.Z())
            particles['vx'].append(vertex.X())
            particles['vy'].append(vertex.Y())
            particles['vz'].append(vertex.Z())
            particles['q'].append(charge)
            particles['nhits'].append(nhits)

            # Loop over all hits
            for hit_info in mcTrackCand.getRelationsWith("RecoHitInformations"):
            
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_VTX and self.UseVTXHits:
                    hit = hit_info.getRelated("VTXClusters")
                    layer = hit.getSensorID().getLayerNumber() - 1
                    used_vtx_hits.append(hit.getArrayIndex())

                    sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
                    position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

                    hits['x'].append(position.X())
                    hits['y'].append(position.Y())
                    hits['z'].append(position.Z())
                    hits['t'].append(0.0)
                    hits['layer'].append(layer)   
                    hits['hit_id'].append(hit_id)
                    hits['particle_id'].append(i)

                    truth['hit_id'].append(hit_id)
                    truth['particle_id'].append(i)
                    truth['weight'].append(0)

                    hit_id += 1

                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_CDC and self.UseCDCHits:
                    hit = hit_info.getRelated("CDCHits")
                    layer = hit.getICLayer() + self.OffsetCDC  
                    used_cdc_hits.append(hit.getArrayIndex())

                    wirePosF = self.cdcGeoHelper.wireForwardPosition(hit.getICLayer(),  hit.getIWire(), 0) 
                    wirePosB = self.cdcGeoHelper.wireBackwardPosition(hit.getICLayer(),  hit.getIWire(), 0) 

                    wireX = 0.5*(wirePosF.X() + wirePosB.X()) 
                    wireY = 0.5*(wirePosF.Y() + wirePosB.Y()) 
                    wireZ = 0.5*(wirePosF.Z() + wirePosB.Z()) 
                    wireT = hit.getTDCCount() 

                    hits['x'].append(wireX)
                    hits['y'].append(wireY)
                    hits['z'].append(wireZ)
                    hits['t'].append(wireT)
                    hits['layer'].append(layer)   
                    hits['hit_id'].append(hit_id)
                    hits['particle_id'].append(i)

                    truth['hit_id'].append(hit_id)
                    truth['particle_id'].append(i)
                    truth['weight'].append(0)

                    hit_id += 1
                   
        # We need to invent mother particle for noise hits
        nid = -1

        particles['particle_id'].append(nid)
        particles['px'].append(0)
        particles['py'].append(0)
        particles['pz'].append(0)
        particles['vx'].append(0.0)
        particles['vy'].append(0.0)
        particles['vz'].append(0.0)
        particles['q'].append(0)
        particles['nhits'].append(0)

        if self.UseVTXHits:
            for hit in vtxClusters:

                if hit.getArrayIndex() in used_vtx_hits:
                    continue

                layer = hit.getSensorID().getLayerNumber() - 1
                sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
                position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

                hits['x'].append(position.X())
                hits['y'].append(position.Y())
                hits['z'].append(position.Z())
                hits['t'].append(0)
                hits['layer'].append(layer)
                hits['hit_id'].append(hit_id)
                hits['particle_id'].append(nid)

                truth['hit_id'].append(hit_id)
                truth['particle_id'].append(nid)
                truth['weight'].append(0)

                hit_id += 1

        if self.UseCDCHits:
            for hit in cdcHits:

                if hit.getArrayIndex() in used_cdc_hits:
                    continue

                layer = hit.getICLayer() + self.OffsetCDC  
            
                wirePosF = self.cdcGeoHelper.wireForwardPosition(hit.getICLayer(),  hit.getIWire(), 0) 
                wirePosB = self.cdcGeoHelper.wireBackwardPosition(hit.getICLayer(),  hit.getIWire(), 0) 

                wireX = 0.5*(wirePosF.X() + wirePosB.X()) 
                wireY = 0.5*(wirePosF.Y() + wirePosB.Y()) 
                wireZ = 0.5*(wirePosF.Z() + wirePosB.Z()) 
                wireT = hit.getTDCCount() 

                hits['x'].append(wireX)
                hits['y'].append(wireY)
                hits['z'].append(wireZ)
                hits['t'].append(wireT)
                hits['layer'].append(layer - 1)
                hits['hit_id'].append(hit_id)
                hits['particle_id'].append(nid)

                truth['hit_id'].append(hit_id)
                truth['particle_id'].append(nid)
                truth['weight'].append(0)

                hit_id += 1

        return pd.DataFrame(hits), pd.DataFrame(truth), pd.DataFrame(particles)