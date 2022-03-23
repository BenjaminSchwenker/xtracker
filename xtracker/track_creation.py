##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

import numpy as np
import torch
from torch_scatter import scatter_add
from torch_scatter import scatter_max
from torch_scatter import scatter_min


def get_degrees(num_nodes, edge_index):
    start, end = torch.LongTensor(edge_index)
    ones = torch.ones(num_nodes)

    in_degree = scatter_add(ones[start], end, dim=0, dim_size=num_nodes)
    out_degree = scatter_add(ones[end], start, dim=0, dim_size=num_nodes)
    return in_degree.data.cpu().numpy(), out_degree.data.cpu().numpy()


def max_edges(num_nodes, edge_index, e):
    start, end = torch.LongTensor(edge_index)
    e = torch.FloatTensor(e)
    ones = torch.ones((num_nodes, 1))

    max_in, argmax_in = scatter_max(e[:, None] * ones[start], end, dim=0, dim_size=num_nodes)
    max_out, argmax_out = scatter_max(e[:, None] * ones[end], start, dim=0, dim_size=num_nodes)
    return argmax_in.squeeze(1).data.cpu().numpy(), argmax_out.squeeze(1).data.cpu().numpy()


def min_edges(num_nodes, edge_index, e):
    start, end = torch.LongTensor(edge_index)
    e = torch.FloatTensor(e)
    ones = torch.ones((num_nodes, 1))

    min_in, argmin_in = scatter_min(e[:, None] * ones[start], end, dim=0, dim_size=num_nodes)
    min_out, argmin_out = scatter_min(e[:, None] * ones[end], start, dim=0, dim_size=num_nodes)
    return argmin_in.squeeze(1).data.cpu().numpy(), argmin_out.squeeze(1).data.cpu().numpy()


def compute_tracks_from_graph_ml(x, edges, preds, cut=0.5, min_hits=3, min_qi=0.0):
    """Compute tracks from output graph [x,edges, preds] following maximum likelyhood
    edges from start node.

    Each row in the tensor x gives node (hit) attributes in the hit graph.  Each column in
    the tensor edges give the index of sender (1st row) and receiver (2nd row) of a directed
    edge in the hitgraph. The tensor pred gives the final segement probabilties for edges.

    It is a slow implementation. For testing purposes only.

    Args:
        x (tensor): node (hit) attributes
        edges (tensor): edge index, subset of segments on hitgraph
        preds (tensor): GNN predictions for segment probabilties
        cut (float): cut on edge (segment) probabilities

    Returns:
        tracks, tracks_qi (list, list): Pair of list giving candidate tracks and their qi
    """

    # Number of nodes (hits)
    num_nodes = x.shape[0]

    # Number of directed edges (segments)
    num_edges = preds.shape[0]

    # For each node, compute the index of the maximum likelyhood
    # ingoing and outgoing edge.
    argmax_in, argmax_out = max_edges(num_nodes, edges, preds)

    # Find seed hits where outgoing track starts
    out_seed_hit_indices = []
    for i in range(num_nodes):
        if argmax_out[i] == num_edges:
            continue

        if argmax_in[i] == num_edges:
            preds_in = 0
        else:
            preds_in = preds[argmax_in[i]]

        if preds[argmax_out[i]] > cut and preds_in < cut:
            out_seed_hit_indices.append(i)

    # Lets try to process the outgoing seeds first.

    # These are the track candidates
    tracks = []

    # These are qi values for track candiates
    tracks_qi = []

    for seed in out_seed_hit_indices:

        curr_hit = seed
        track = [curr_hit, ]
        qi = 0.0

        for step in range(400):
            # Check first stop condition
            if argmax_out[curr_hit] == num_edges:
                break

            delta_qi = preds[argmax_out[curr_hit]]

            # Check 2nd stop condition
            if delta_qi < cut:
                break

            # Move to next hit
            curr_hit = edges[1, argmax_out[curr_hit]]
            track.append(curr_hit)
            qi += delta_qi

        if len(track) >= min_hits:
            qi = qi / (len(track) - 1)

        if len(track) >= min_hits and qi >= min_qi:
            tracks.append(track)
            tracks_qi.append(qi)

    return tracks, tracks_qi
