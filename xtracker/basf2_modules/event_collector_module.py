##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

from ROOT import Belle2
import basf2 as b2
import pandas as pd

from xtracker.event_creation import make_event_with_mc


class TrackingEventCollector(b2.Module):
    """Module to collect tracking information per event and write it to an output directory. The set of
    tracking events will later be used to train tracking algorithms."""

    def __init__(
        self,
        event_cuts,
        output_dir_name,
        cdcHitsColumnName='CDCHits',
        trackCandidatesColumnName="RecoTracks",
        mcTrackCandidatesColumName="MCRecoTracks",
    ):
        """Constructor"""

        super(TrackingEventCollector, self).__init__()

        #: cached dictionary of cuts defining event data
        self.event_cuts = event_cuts
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

        cdcHits = Belle2.PyStoreArray(self.cdcHitsColumnname)
        vtxClusters = Belle2.PyStoreArray("VTXClusters")
        mcTrackCands = Belle2.PyStoreArray(self.mcTrackCandidatesColumnName)

        hits, truth, particles, detector_info = make_event_with_mc(
            cdcHits, vtxClusters, self.trackMatchLookUp, mcTrackCands, self.event_cuts)

        filename = self.output_dir_name + '/{}_id_{}.h5'.format("event", evtid)
        hits.to_hdf(filename, key='hits')
        truth.to_hdf(filename, key='truth')
        particles.to_hdf(filename, key='particles')
