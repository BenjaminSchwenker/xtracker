# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


import numpy as np
import basf2 as b2
from ROOT import Belle2

from xtracker.utils import dotdict
from xtracker.gnn_tracking.TrackingGame import TrackingGame as Game
from xtracker.gnn_tracking.pytorch.NNet import NNetWrapper
from xtracker.gnn_tracking.pytorch.NNet import NNetWrapperTrigger
from xtracker.gnn_tracking.ImTracker import ImTracker
from xtracker.datasets.graph import graph_to_sparse, get_batch
from xtracker.event_creation import make_event
from xtracker.graph_creation import make_graph


class GNNTrigger(b2.Module):
    """Module to find tracks from hits in the tracking detector using graph neural network tracking."""

    def __init__(
        self,
        tracker_model_path,
        trigger_model_path,
        event_cuts,
        segment_cuts,
        tracker_config,
        networks_config,
        threshold=0.5,
        cdcHitsColumnName='CDCHits',
    ):
        """Constructor"""

        super(GNNTrigger, self).__init__()

        #: cached value of tracker model path
        self.tracker_model_path = tracker_model_path
        #: cached value of trigger model path
        self.trigger_model_path = trigger_model_path
        #: cached dictionary of cuts defining event data
        self.event_cuts = event_cuts
        #: cached dictionary of cuts for segment selection
        self.segment_cuts = segment_cuts
        #: cached dictionary of params for trackfinding
        self.tracker_config = tracker_config
        #: cached dictionary of params for GNN networks
        self.networks_config = networks_config
        #: cached name of the CDCHits StoreArray
        self.cdcHitsColumnname = cdcHitsColumnName
        #: cached value of trigger threshold
        self.threshold = threshold

    def initialize(self):
        """Receive signal at the start of event processing"""

        #: cached tracking game object
        self.game = Game(train_data_loader=None, valid_data_loader=None)

        #: cached tracker object
        self.tracker = self.setupTracker_()

        #: cached trigger object
        self.trigger = self.setupTrigger_()

        #: cached extra info to hold trigger decision
        self.event_extra_info = Belle2.PyStoreObj('EventExtraInfo')

        self.event_extra_info.registerInDataStore()

    def setupTracker_(self):
        n1 = NNetWrapper(self.networks_config['embedding_dim'],self.networks_config['tracker_layer_size'],self.networks_config['n_update_iters'])
        n1.load_checkpoint(self.tracker_model_path, 'best.pth.tar')
        tracker = ImTracker(self.game, n1, dotdict(self.tracker_config))
        return tracker

    def setupTrigger_(self):
        trigger = NNetWrapperTrigger(self.networks_config['embedding_dim'],self.networks_config['trigger_layer_size'])
        trigger.load_checkpoint(self.trigger_model_path, 'best.pth.tar')
        return trigger

    def event(self):
        """Event method"""

        event_meta_data = Belle2.PyStoreObj("EventMetaData")
        evtid = event_meta_data.getEvent()

        vtxClusters = Belle2.PyStoreArray("VTXClusters")
        cdcHits = Belle2.PyStoreArray(self.cdcHitsColumnname)

        # 1) Create event data
        hits, truth, particles, detector_info, trigger = make_event(cdcHits, vtxClusters, self.event_cuts)

        # 2) Make hit graph for event data
        graphs, IDs = make_graph(
            hits,
            truth,
            particles,
            trigger,
            evtid,
            **self.segment_cuts,
            phi_range=(-np.pi, np.pi),
        )

        # 3) Compute the trigger
        batch = get_batch(graph_to_sparse(graphs[0]))
        board = self.game.getInitBoardFromGraph(batch)

        x = self.tracker.embed_hits(board)
        pred_trig = self.trigger.predict(x)[0, 0]

        # 4) Save trigger decision in ExtraEventInfo
        if not self.event_extra_info.isValid():
            self.event_extra_info.create()
        else:
            if self.event_extra_info.hasExtraInfo('VTXTrigger'):
                b2.B2FATAL('The EventExtraInfo object has already an VTXTrigger field registered.')

        self.event_extra_info.setExtraInfo('VTXTriggerClassifierOutput', pred_trig)
        self.event_extra_info.setExtraInfo('VTXTrigger', pred_trig > self.threshold)
