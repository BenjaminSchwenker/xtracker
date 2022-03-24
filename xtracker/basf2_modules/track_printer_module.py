# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


import numpy as np
import math
from ROOT import Belle2
from ROOT import TVector3
import basf2 as b2

from tracking.validation.utilities import (
    getHelixFromMCParticle,
    is_primary,
    get_det_hit_ids,
    calc_ndf_from_det_hit_ids,
)


class TrackPrinter(b2.Module):
    """Module to print track contents to stdout."""

    def __init__(
        self,
        cdcHitsColumnName='CDCHits',
        trackCandidatesColumnName="RecoTracks",
        mcTrackCandidatesColumName="MCRecoTracks",
        printSimHits=True,
    ):
        """Constructor"""

        super(TrackPrinter, self).__init__()

        self.cdcHitsColumnname = cdcHitsColumnName
        #: cached name of the RecoTracks StoreArray
        self.trackCandidatesColumnName = trackCandidatesColumnName
        #: cached name of the MCRecoTracks StoreArray
        self.mcTrackCandidatesColumnName = mcTrackCandidatesColumName
        #: cached flag for printing simhits
        self.printSimHits = printSimHits

    def initialize(self):
        """Receive signal at the start of event processing"""

        #: Track-match object that examines relation information from MCMatcherTracksModule
        self.trackMatchLookUp = Belle2.TrackMatchLookUp(self.mcTrackCandidatesColumnName, self.trackCandidatesColumnName)

    def event(self):
        """Event method"""

        trackMatchLookUp = self.trackMatchLookUp

        print('Print {}:'.format(self.mcTrackCandidatesColumnName))

        # Analyse from the Monte Carlo reference side
        mcTrackCands = Belle2.PyStoreArray(self.mcTrackCandidatesColumnName)

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

            print('  {} MCTrack nhits={} charge={} pt={} and tanLambda={}'.format(i, nhits, charge, pt, tan_lambda))

            ihit = 0
            # Loop over all hits
            for hit_info in mcTrackCand.getRelationsWith("RecoHitInformations"):

                ihit += 1
                layer = np.float("nan")
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_SVD:
                    hit = hit_info.getRelated("SVDClusters")
                    layer = hit.getSensorID().getLayerNumber()
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_PXD:
                    hit = hit_info.getRelated("PXDClusters")
                    layer = hit.getSensorID().getLayerNumber()
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_VTX:
                    hit = hit_info.getRelated("VTXClusters")
                    layer = hit.getSensorID().getLayerNumber()

                    if self.printSimHits:
                        simHit = hit.getRelated('VTXTrueHits')
                        if not simHit:
                            print("     Skipping VTXCluster without related VTXTrueHit.")
                            print("     This should not happen")
                            continue

                        tof = simHit.getGlobalTime()
                        print('     {} hit tof={}, layer={}'.format(ihit, tof, layer))

                    else:
                        sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
                        position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

                        print('     {} hit x={}, y={} z={}, layer={}'.format(ihit, position.X(), position.Y(), position.Z(), layer))

                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_CDC:
                    hit = hit_info.getRelated("CDCHits")
                    layer = hit.getICLayer()

                    simHit = hit.getRelated('CDCSimHits')
                    if not simHit:
                        print("     Skipping CDCHit without related CDCSimHit.")
                        print("     This should not happen")
                        continue

                    tof = simHit.getFlightTime()

                    print('     {} cdchit tof={} layer={}'.format(ihit, tof, layer))

        print('Print {}:'.format(self.trackCandidatesColumnName))

        # Analyse from the PR side
        prTrackCands = Belle2.PyStoreArray(self.trackCandidatesColumnName)

        for i, prTrackCand in enumerate(prTrackCands):

            qi = prTrackCand.getQualityIndicator()
            det_hit_ids = get_det_hit_ids(prTrackCand)
            ndf = calc_ndf_from_det_hit_ids(det_hit_ids)
            nhits = len(det_hit_ids)

            print('  {} PRTrack has nhits={} qi={}'.format(i, nhits, qi))

            ihit = 0
            # Loop over all hits
            for hit_info in prTrackCand.getRelationsWith("RecoHitInformations"):

                ihit += 1
                layer = np.float("nan")
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_SVD:
                    hit = hit_info.getRelated("SVDClusters")
                    layer = hit.getSensorID().getLayerNumber()
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_PXD:
                    hit = hit_info.getRelated("PXDClusters")
                    layer = hit.getSensorID().getLayerNumber()
                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_VTX:
                    hit = hit_info.getRelated("VTXClusters")
                    layer = hit.getSensorID().getLayerNumber()

                    if self.printSimHits:
                        simHit = hit.getRelated('VTXTrueHits')
                        if not simHit:
                            print("     Skipping VTXCluster without related VTXTrueHit.")
                            print("     This should not happen")
                            continue

                        tof = simHit.getGlobalTime()

                        print('     {} hit tof={}, layer={}'.format(ihit, tof, layer))

                    else:
                        sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
                        position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

                        print('     {} hit x={}, y={} z={}, layer={}'.format(ihit, position.X(), position.Y(), position.Z(), layer))

                if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_CDC:
                    hit = hit_info.getRelated("CDCHits")
                    layer = hit.getICLayer()

                    simHit = hit.getRelated('CDCSimHits')
                    if not simHit:
                        print("     Skipping CDCHit without related CDCSimHit.")
                        print("     This should not happen")
                        continue

                    tof = simHit.getFlightTime()

                    print('     {} cdchit  tof={}, layer={}'.format(ihit, tof, layer))
