# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

import numpy as np
import pandas as pd
import math
import collections

from ROOT import Belle2
from ROOT import TVector3


from tracking.validation.utilities import (
    getHelixFromMCParticle,
    get_det_hit_ids,
    calc_ndf_from_det_hit_ids,
)


def calc_dphi(phi1, phi2):
    """Computes abs(phi2-phi1) given in range [-pi,pi]. NAN inputs produce NAN outputs."""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2 * np.pi
    if dphi < -np.pi:
        dphi += 2 * np.pi
    return abs(dphi)


def make_empty_event():
    """Returns empty event data tables for xtracker."""

    # Initialze variables to be returned
    hits = {'particle_id': [], 'layer': [], 'x': [], 'y': [], 'z': [], 't': [], 'hit_id': []}
    truth = {'hit_id': [], 'particle_id': [], 'weight': []}
    particles = {'vx': [], 'vy': [], 'vz': [], 'px': [], 'py': [], 'pz': [], 'q': [], 'nhits': [], 'particle_id': []}
    detector_info = []

    return hits, truth, particles, detector_info


def make_event(cdcHits, vtxClusters, event_cuts):
    """
    Returns event data from basf2 store arrays with reconstructed hits in tracking detectors.

    No MC truth information is used.
    """

    # Create an empty event structure
    hits, truth, particles, detector_info = make_empty_event()

    # Invent a dummy mother particle for hits, we do not use MC
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

    if event_cuts['UseVTXHits']:

        for hit in vtxClusters:

            layer = hit.getSensorID().getLayerNumber() + event_cuts['OffsetVTX']
            sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
            position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

            detector_info.append((Belle2.RecoHitInformation.c_VTX, hit.getArrayIndex()))

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

    if event_cuts['UseCDCHits']:

        for hit in cdcHits:

            layer = hit.getICLayer() + event_cuts['OffsetCDC']
            detector_info.append((Belle2.RecoHitInformation.c_CDC, hit.getArrayIndex()))

            WireHit = Belle2.TrackFindingCDC.CDCWireHit(hit)
            wirePos = WireHit.getRefPos3D()

            wireX = wirePos.x()
            wireY = wirePos.y()
            wireZ = wirePos.z()
            wireT = hit.getTDCCount()

            hits['x'].append(wireX)
            hits['y'].append(wireY)
            hits['z'].append(wireZ)
            hits['t'].append(wireT)
            hits['layer'].append(layer)
            hits['hit_id'].append(hit_id)
            hits['particle_id'].append(nid)

            truth['hit_id'].append(hit_id)
            truth['particle_id'].append(nid)
            truth['weight'].append(0)

            hit_id += 1

    return pd.DataFrame(hits), pd.DataFrame(truth), pd.DataFrame(particles), detector_info


def make_event_with_mc(cdcHits, vtxClusters, trackMatchLookUp, mcTrackCands, event_cuts):
    """
    Returns event data from basf2 store arrays with reconstructed hits in tracking detectors.

    This function also uses MC truth information from the store array mcTrackCands. The array
    mcTrackCands contains basf2 RecoTracks from the MC truth track finder. It contains a flight
    time sorted list of reco hits from tracking subdetectors for each  RecoTrack from truth
    track finder. Reco hits are further linked to SimHits. The trackmatchlookup object allow
    accessing the MCParticle related to a given track.
    """

    # Create an empty event data structure
    hits, truth, particles, detector_info = make_empty_event()

    if not mcTrackCands:
        return pd.DataFrame(hits), pd.DataFrame(truth), pd.DataFrame(particles), detector_info

    used_vtx_hits = []
    used_cdc_hits = []

    # Running counter used to enumerate all hits in this event
    hit_id = 0

    for i, mcTrackCand in enumerate(mcTrackCands):

        mcParticle = trackMatchLookUp.getRelatedMCParticle(mcTrackCand)
        mcHelix = getHelixFromMCParticle(mcParticle)

        momentum = mcParticle.getMomentum()
        vertex = mcParticle.getProductionVertex()
        charge = mcParticle.getCharge()
        time = mcParticle.getProductionTime()

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

        # Buffer to keep previous two cdc sim hits
        cdcHitBuffer = collections.deque(maxlen=2)
        last_valid = False
        last_track_pos = np.nan
        last_track_mom = np.nan

        # Loop over all hits
        for hit_info in mcTrackCand.getRelationsWith("RecoHitInformations"):

            if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_VTX and event_cuts['UseVTXHits']:
                hit = hit_info.getRelated("VTXClusters")
                layer = hit.getSensorID().getLayerNumber() + event_cuts['OffsetVTX']
                sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
                position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

                simHit = hit.getRelated('VTXTrueHits')
                if not simHit:
                    print("Skipping VTXCluster without related VTXTrueHit. This should not happen.")
                    continue

                tof = simHit.getGlobalTime()

                delta = tof - time
                if delta < -0.01:
                    print('Skipping VTXCluster because of negative delta tof={}'.format(delta))
                    continue

                used_vtx_hits.append(hit.getArrayIndex())
                detector_info.append((Belle2.RecoHitInformation.c_VTX, hit.getArrayIndex()))

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

                time = tof
                hit_id += 1

            if hit_info.getTrackingDetector() == Belle2.RecoHitInformation.c_CDC and event_cuts['UseCDCHits']:

                hit = hit_info.getRelated("CDCHits")
                layer = hit.getICLayer() + event_cuts['OffsetCDC']

                simHit = hit.getRelated('CDCSimHits')
                if not simHit:
                    print("Skipping CDCHit without related CDCSimHit from track. This should not happen.")
                    continue

                tof = simHit.getFlightTime()

                delta = tof - time
                if delta < -0.01:
                    print("""Skipping CDC hit from track because too negative delta time
                        of flight: layer={} delta_tof={}""".format(layer, delta)
                          )
                    continue

                if delta > 1:
                    print('Skipping all remaining hits because delta tof is very big: delta={}>1ns'.format(delta))
                    break

                simHitPos = simHit.getPosTrack()
                simHitMom = simHit.getMomentum()

                if len(cdcHitBuffer) > 1:
                    d1 = simHitPos - cdcHitBuffer[-1]
                    d2 = cdcHitBuffer[-1] - cdcHitBuffer[-2]

                    cosTheta = d1.Dot(d2) / d2.Mag() / d1.Mag()
                    if cosTheta < 0 and d1.Mag() > 0.1:
                        print("""Skipping CDC hit from track because successive simhit momenta
                            have negative cosTheta={}""".format(cosTheta)
                              )
                        continue

                    if d1.Mag() > 20:
                        print('Skipping all remaining hits because very long step length={}>20cm'.format(d1.Mag()))
                        break

                WireHit = Belle2.TrackFindingCDC.CDCWireHit(hit)
                wirePos = WireHit.getRefPos3D()

                wireX = wirePos.x()
                wireY = wirePos.y()
                wireZ = wirePos.z()
                wireT = hit.getTDCCount()

                used_cdc_hits.append(hit.getArrayIndex())
                detector_info.append((Belle2.RecoHitInformation.c_CDC, hit.getArrayIndex()))

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

                cdcHitBuffer.append(simHitPos)
                last_valid = True
                last_track_pos = simHitPos
                last_track_mom = simHitMom

                time = tof
                hit_id += 1

    # We need to look at all remaining hits now. They are noise. Invent mother particle for noise hits
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

    if event_cuts['UseVTXHits']:
        for hit in vtxClusters:

            if hit.getArrayIndex() in used_vtx_hits:
                continue

            layer = hit.getSensorID().getLayerNumber() + event_cuts['OffsetVTX']
            sensor_info = Belle2.VXD.GeoCache.get(hit.getSensorID())
            position = sensor_info.pointToGlobal(TVector3(hit.getU(), hit.getV(), 0), True)

            detector_info.append((Belle2.RecoHitInformation.c_VTX, hit.getArrayIndex()))

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

    if event_cuts['UseCDCHits']:

        for hit in cdcHits:

            if hit.getArrayIndex() in used_cdc_hits:
                continue

            layer = hit.getICLayer() + event_cuts['OffsetCDC']
            detector_info.append((Belle2.RecoHitInformation.c_CDC, hit.getArrayIndex()))

            WireHit = Belle2.TrackFindingCDC.CDCWireHit(hit)
            wirePos = WireHit.getRefPos3D()

            wireX = wirePos.x()
            wireY = wirePos.y()
            wireZ = wirePos.z()
            wireT = hit.getTDCCount()

            hits['x'].append(wireX)
            hits['y'].append(wireY)
            hits['z'].append(wireZ)
            hits['t'].append(wireT)
            hits['layer'].append(layer)
            hits['hit_id'].append(hit_id)
            hits['particle_id'].append(nid)

            truth['hit_id'].append(hit_id)
            truth['particle_id'].append(nid)
            truth['weight'].append(0)

            hit_id += 1

    return pd.DataFrame(hits), pd.DataFrame(truth), pd.DataFrame(particles), detector_info
