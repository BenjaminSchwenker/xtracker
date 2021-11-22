##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

import numpy as np


def compute_tracks_from_graph(x, edges, preds, cut=0.5, min_mc_hits=3):
    """Compute tracks from segment classifier outputs using a cut approach

    The tracks are computed as connected components of event_graph. The event graph
    connects hits only when the edge probability exceeds a cut.

    Args:
        edges: edge index, subset of segments on hitgraph
        preds: GNN predictions on edge index

    Returns:
        tracks: networkx.connected_components retun value
    """

    num_nodes = x.shape[0]
    num_edges = preds.shape[0]

    tracks = []

    for j in range(num_edges):

        if preds[j] > cut:

            sp1, sp2 = edges[0, j], edges[1, j]

            isNewTrack = True
            for trk in tracks:
                if sp1 in trk:
                    isNewTrack = False
                    trk.append(sp2)
                    break
                elif sp2 in trk:
                    isNewTrack = False
                    trk.append(sp1)
                    break

            if isNewTrack:
                tracks.append([sp1, sp2])

    tracks = [list(set(track)) for track in tracks]

    tracks = [track for track in tracks if len(track) >= min_mc_hits]

    return tracks
