##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################


import numpy as np
import pandas as pd

from xtracker.datasets.graph import Graph


def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi
    return dphi


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))


def select_segments(hits1, hits2, phi_slope_max, z0_max):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    # Start with all possible pairs of hits
    keys = ['evtid', 'r', 'phi', 'z', 'layer']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))
    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr

    # We do not have good filter for same layer or ingoing edges
    z0[hit_pairs.layer_1 >= hit_pairs.layer_2] = 0.0
    phi_slope[hit_pairs.layer_1 == hit_pairs.layer_2] = 0.0

    # Compute squared 'distance' to exclude self loops
    dr2 = dz**2 + dr**2 + dphi**2

    # Filter segments according to criteria
    good_seg_mask = ((phi_slope > -phi_slope_max) &
                     (phi_slope < phi_slope_max) &
                     (z0 > -z0_max) &
                     (z0 < z0_max) &
                     (dr2 > 0)
                     )

    return hit_pairs[['index_1', 'index_2']][good_seg_mask]


def make_graph(
    hits,
    truth,
    particles,
    evtid,
    n_det_layers,
    pt_min,
    phi_range,
    n_phi_sections,
    eta_range,
    n_eta_sections,
    segment_type,
    z0_max,
    phi_slope_max,
    feature_scale_r,
    feature_scale_phi,
    feature_scale_z,
):

    # Apply hit selection
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(evtid=evtid)

    # Divide detector into sections
    phi_edges = np.linspace(*phi_range, num=n_phi_sections + 1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections + 1)
    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)

    # Graph features and scale
    feature_names = ['r', 'phi', 'z']

    # Scale hit coordinates
    feature_scale = np.array([feature_scale_r, np.pi / n_phi_sections / feature_scale_phi, feature_scale_z])

    layer_pairs = form_layer_pairs(n_det_layers, segment_type)

    # Construct the graph
    graphs_all = [construct_graph(section_hits, layer_pairs=layer_pairs,
                                  phi_slope_max=phi_slope_max, z0_max=z0_max,
                                  feature_names=feature_names,
                                  feature_scale=feature_scale)
                  for section_hits in hits_sections]
    graphs = [x[0] for x in graphs_all]
    IDs = [x[1] for x in graphs_all]

    return graphs, IDs


def construct_segments(hits, layer_pairs, phi_slope_max, z0_max):
    # Loop over layer pairs and construct segments
    layer_groups = hits.groupby('layer')
    segments = []
    for (layer1, layer2) in layer_pairs:
        # Find and join all hit pairs
        try:
            hits1 = layer_groups.get_group(layer1)
            hits2 = layer_groups.get_group(layer2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            continue
        # Construct the segments
        segments.append(select_segments(hits1, hits2, phi_slope_max, z0_max))
    # Combine segments from all layer pairs
    if len(segments) == 0:
        return pd.DataFrame(columns=['index_1', 'index_2'])
    else:
        return pd.concat(segments)


def construct_graph(hits, layer_pairs,
                    phi_slope_max, z0_max,
                    feature_names, feature_scale):
    """Construct one graph (e.g. from one event)"""

    segments = construct_segments(hits, layer_pairs, phi_slope_max, z0_max)

    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    X = (hits[feature_names].values / feature_scale).astype(np.float32)
    P = (hits[['px', 'py', 'pz', 'q', 'nhits', 'particle_id']].values).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y = np.zeros(n_edges, dtype=np.float32)
    hitId = hits['hit_id']

    # We have the segments' hits given by dataframe label,
    # so we need to translate into positional indices.
    # Use a series to map hit label-index onto positional-index.
    hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
    seg_start = hit_idx.loc[segments.index_1].values
    seg_end = hit_idx.loc[segments.index_2].values

    # Now we can fill the association matrices.
    # Note that Ri maps hits onto their incoming edges,
    # which are actually segment endings.
    Ri[seg_end, np.arange(n_edges)] = 1
    Ro[seg_start, np.arange(n_edges)] = 1
    # Fill the segment labels
    pid1 = hits.particle_id.loc[segments.index_1].values
    pid2 = hits.particle_id.loc[segments.index_2].values
    hid1 = hits.hit_id.loc[segments.index_1].values
    hid2 = hits.hit_id.loc[segments.index_2].values

    # Connect only hits from same particle and consecutive
    # in flight direction (sorted by hit_id)
    y[:] = (pid1 == pid2) & (hid1 + 1 == hid2) & (pid1 >= 0) & (pid2 >= 0)

    # Return a tuple of the results
    return Graph(X, Ri, Ro, y, P), hitId


def select_hits(hits, truth, particles, pt_min=0):

    # Calculate particle transverse momentum
    pt = np.sqrt(particles.px**2 + particles.py**2)
    # True particle selection.
    # Applies pt cut, removes all noise hits.
    particles = particles[pt >= pt_min]
    truth = (truth[['hit_id', 'particle_id']]
             .merge(particles[['particle_id', 'px', 'py', 'pz', 'q', 'nhits']], on='particle_id'))
    # Calculate derived hits variables
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    # Select the data columns we need
    hits = (hits[['hit_id', 'z', 'layer']]
            .assign(r=r, phi=phi)
            .merge(truth[['hit_id', 'particle_id', 'px', 'py', 'pz', 'q', 'nhits']], on='hit_id'))

    return hits


def split_detector_sections(hits, phi_edges, eta_edges):
    """Split hits according to provided phi and eta boundaries."""
    hits_sections = []
    # Loop over sections
    for i in range(len(phi_edges) - 1):
        phi_min, phi_max = phi_edges[i], phi_edges[i + 1]
        # Select hits in this phi section
        phi_hits = hits[(hits.phi > phi_min) & (hits.phi < phi_max)]
        # Center these hits on phi=0
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2
        phi_hits = phi_hits.assign(phi=centered_phi, phi_section=i)
        for j in range(len(eta_edges) - 1):
            eta_min, eta_max = eta_edges[j], eta_edges[j + 1]
            # Select hits in this eta section
            eta = calc_eta(phi_hits.r, phi_hits.z)
            sec_hits = phi_hits[(eta > eta_min) & (eta < eta_max)]
            hits_sections.append(sec_hits.assign(eta_section=j))
    return hits_sections


def form_layer_pairs(n_det_layers, segment_type):

    # Define adjacent layers
    layerIDs = np.arange(n_det_layers)
    if segment_type == 'out':
        # Create only outgoing segments
        layer_pairs = np.stack([layerIDs[:-1], layerIDs[1:]], axis=1)
    elif segment_type == 'inout':
        # Create ingoing and outgoing segments
        l_1 = np.concatenate((layerIDs[:-1], layerIDs[1:]), axis=None)
        l_2 = np.concatenate((layerIDs[1:], layerIDs[:-1]), axis=None)
        layer_pairs = np.stack((l_1, l_2), axis=-1)
    elif segment_type == 'all':
        # Create ingoing and outgoing segments and segements on same layer
        # This is needed for fully reconstructing loopers
        l_1 = np.concatenate((layerIDs[:-1], layerIDs[1:], layerIDs[:]), axis=None)
        l_2 = np.concatenate((layerIDs[1:], layerIDs[:-1], layerIDs[:]), axis=None)
        layer_pairs = np.stack((l_1, l_2), axis=-1)

    return layer_pairs
