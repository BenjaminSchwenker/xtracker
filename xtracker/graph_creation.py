# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


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
    """Compute rapidity eta from radius and z position of a point"""
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))


def create_segments_mc(hits1, hits2):
    """
    Construct a list of segments from the pairings between hits1 and
    hits2 along a the flight time sorted path of a mc particle.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """

    # Combine sorted hits into a hit_pairs data frame
    keys = ['evtid', 'r', 'phi', 'z', 'layer']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), left_index=True, right_index=True, suffixes=('_1', '_2'))

    return hit_pairs


def create_segments(hits1, hits2):
    """
    Construct a list of raw (unfiltered) segments from the pairings
    between hits1 and hits2.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """

    # Start with all possible pairs of hits
    keys = ['evtid', 'r', 'phi', 'z', 'layer']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))

    return hit_pairs


def cut_on_segments(segments, phi_slope_max, z0_max, debug=False):
    """
    Cut on segments (hit pairs) by phi slope, z0 and other criteria.

    For MC truth segments, any removed segments will reduce the hit efficiency
    and will create clone tracks. For debugging, need to report number of
    removed true segments.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """

    dphi = calc_dphi(segments.phi_1, segments.phi_2)
    dz = segments.z_2 - segments.z_1
    dr = segments.r_2 - segments.r_1
    dr2 = dz**2 + dr**2 + dphi**2
    dphi_abs = dphi.abs()

    phi_slope = dphi / dr
    phi_slope[segments.layer_1 == segments.layer_2] = 0.0

    z0 = segments.z_1 - segments.r_1 * dz / dr
    z0[segments.layer_1 >= segments.layer_2] = 0.0

    layer_diff = (segments.layer_1 - segments.layer_2).abs()

    # 0) Apply global cuts on segments, not subdetector specific

    # Hits should differ in at least one attribute to avoid self loops
    good_seg_mask = (
        (layer_diff < 4) &
        (dr2 > 0)
    )

    # 1) Apply filters that are specific to segments between vertex layers
    vxd_segments = (segments.layer_1 < 5) & (segments.layer_2 < 5)

    vxd_mask = (
        (phi_slope > -phi_slope_max) &
        (phi_slope < phi_slope_max) &
        (z0 > -z0_max) &
        (z0 < z0_max)
    )
    vxd_mask[~vxd_segments] = True
    good_seg_mask = good_seg_mask & vxd_mask

    # 2) Apply filters that are specific to segments between cdc layers
    cdc_segments = (segments.layer_1 >= 5) & (segments.layer_2 >= 5)
    cdc_mask = (
        (dphi_abs < 0.2)
    )

    cdc_mask[~cdc_segments] = True
    good_seg_mask = good_seg_mask & cdc_mask

    # 3) Apply filters that are specific to segments cdc and vtx
    vtx_cdc_segments = (
        ((segments.layer_1 <= 4) & (segments.layer_2 >= 5)) | ((segments.layer_1 >= 5) & (segments.layer_2 <= 4))
    )

    vtx_cdc_mask = (
        (phi_slope > -phi_slope_max) &
        (phi_slope < phi_slope_max)
    )

    vtx_cdc_mask[~vtx_cdc_segments] = True
    good_seg_mask = good_seg_mask & vtx_cdc_mask

    # add some reporting in debug mode
    debug = False
    if debug and good_seg_mask.shape[0] - good_seg_mask.sum() > 0:
        print('All segements: ', good_seg_mask.shape[0])
        print('Removed segements: ', good_seg_mask.shape[0] - good_seg_mask.sum())

        i = np.random.randint(100)

        draw_sample_xy(segments, good_seg_mask, i, figsize=(9, 9))
        draw_sample_rz(segments, good_seg_mask, i, figsize=(9, 9))

    return segments[['index_1', 'index_2']][good_seg_mask]


def draw_sample_xy(segments, good_segments, i, figsize=(9, 9)):

    import matplotlib.pyplot as plt

    x_1 = segments.r_1 * np.cos(segments.phi_1)
    x_2 = segments.r_2 * np.cos(segments.phi_2)

    y_1 = segments.r_1 * np.sin(segments.phi_1)
    y_2 = segments.r_2 * np.sin(segments.phi_2)

    fig, ax0 = plt.subplots(figsize=figsize)

    # Draw the hits
    ax0.scatter(x_1, y_1, s=2, c='k')
    ax0.scatter(x_2, y_2, s=2, c='k')

    # Draw the segments
    for iseg in range(len(segments)):

        xp = np.array([x_1[iseg], x_2[iseg]])
        yp = np.array([y_1[iseg], y_2[iseg]])

        # Only draw true hit hitgraph

        if good_segments[iseg]:
            ax0.plot(xp, yp, '-', c='g')
        else:
            ax0.plot(xp, yp, '--', c='b')

    plt.savefig("/home/benjamin/Desktop/gen_study/{}_xy.png".format(i))


def draw_sample_rz(segments, good_segments, i, figsize=(9, 9)):

    import matplotlib.pyplot as plt

    y_1 = segments.r_1
    y_2 = segments.r_2

    x_1 = segments.z_1
    x_2 = segments.z_2

    fig, ax0 = plt.subplots(figsize=figsize)

    # Draw the hits
    ax0.scatter(x_1, y_1, s=2, c='k')
    ax0.scatter(x_2, y_2, s=2, c='k')

    # Draw the segments
    for iseg in range(len(segments)):

        xp = np.array([x_1[iseg], x_2[iseg]])
        yp = np.array([y_1[iseg], y_2[iseg]])

        # Only draw true hit hitgraph

        if good_segments[iseg]:
            ax0.plot(xp, yp, '-', c='g')
        else:
            ax0.plot(xp, yp, '--', c='b')

    plt.savefig("/home/benjamin/Desktop/gen_study/{}_rz.png".format(i))


def make_graph(
    hits,
    truth,
    particles,
    trigger,
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
    feature_scale_t,
    useMC=False,
):
    """Returns a graph object and a hitID array computed from event data."""

    # Apply hit selection
    hits = select_hits(hits, truth, particles, pt_min=pt_min).assign(evtid=evtid)

    # Divide detector into sections
    phi_edges = np.linspace(*phi_range, num=n_phi_sections + 1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sections + 1)
    hits_sections = split_detector_sections(hits, phi_edges, eta_edges)

    # Graph features and scale
    feature_names = ['r', 'phi', 'z']

    # Scale hit coordinates
    feature_scales = np.array([feature_scale_r, np.pi / n_phi_sections / feature_scale_phi, feature_scale_z])

    if useMC:
        # Construct the graph only from mc (truth) level data
        graphs_all = [construct_graph_mc(
            section_hits, truth,
            feature_names=feature_names,
            feature_scales=feature_scales,
            trigger=trigger
        ) for section_hits in hits_sections]

    else:
        # Construct the graph only from reconstruction level data
        graphs_all = [construct_graph(
            section_hits, n_det_layers, segment_type,
            phi_slope_max=phi_slope_max, z0_max=z0_max,
            feature_names=feature_names,
            feature_scales=feature_scales,
            trigger=trigger
        ) for section_hits in hits_sections]

    graphs = [x[0] for x in graphs_all]
    IDs = [x[1] for x in graphs_all]

    return graphs, IDs


def construct_segments(hits, layer_pairs, phi_slope_max, z0_max):
    """Returns DataFrame with filtered segments (hit pairs) for event."""

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
        except KeyError:
            continue
        # Construct the segments
        raw_segments = create_segments(hits1, hits2)

        # Cut on good segments and append them
        segments.append(cut_on_segments(raw_segments, phi_slope_max=phi_slope_max, z0_max=z0_max))

    # Combine segments from all layer pairs
    if len(segments) == 0:
        return pd.DataFrame(columns=['index_1', 'index_2'])
    else:
        return pd.concat(segments)


def construct_segments_mc(hits, truth, phi_slope_max=np.inf, z0_max=np.inf):
    """Returns DataFrame with filtered segments (hit pairs) for event using MC truth labels."""

    # Only connect hits from mcparticles
    mask = truth['particle_id'] >= 0

    # Group hits from same mcparticle
    grouped = truth[mask].groupby("particle_id")

    segments = []
    for name, group in grouped:
        pairs = group[['hit_id']].astype(np.float)
        pairs.rename(columns={pairs.columns[0]: "index_0"}, inplace=True)
        pairs["index_1"] = pairs["index_0"].shift(-1)
        pairs = pairs.iloc[:-1]
        pairs = pairs.astype(np.int64)

        hits1 = hits.iloc[pairs['index_0']]
        hits2 = hits.iloc[pairs['index_1']]

        # Construct the segments
        raw_segments = create_segments_mc(hits1, hits2)

        # Cut on good segments and append them
        segments.append(cut_on_segments(raw_segments, phi_slope_max=phi_slope_max, z0_max=z0_max, debug=True))

    # Combine segments from all layer pairs
    if len(segments) == 0:
        return pd.DataFrame(columns=['index_1', 'index_2'])
    else:
        return pd.concat(segments)


def construct_graph(hits, n_det_layers, segment_type,
                    phi_slope_max, z0_max,
                    feature_names, feature_scales, trigger):
    """Construct one graph (e.g. from one event)"""

    # Construct layer pairs
    layer_pairs = form_layer_pairs(n_det_layers, segment_type)

    skip2_pairs = form_skip_layer_pairs(first_layer_id=4, last_layer_id=60, skip=2, segment_type='all')
    layer_pairs = np.concatenate((layer_pairs, skip2_pairs), axis=0)

    skip3_pairs = form_skip_layer_pairs(first_layer_id=4, last_layer_id=60, skip=3, segment_type='all')
    layer_pairs = np.concatenate((layer_pairs, skip3_pairs), axis=0)

    # Construct filtered segments
    segments = construct_segments(hits, layer_pairs, phi_slope_max, z0_max)

    # Prepare the graph tuple with selected and scaled hit features
    graph = prepare_graph_matrices(hits, segments, feature_names, feature_scales, trigger)

    return graph


def construct_graph_mc(hits, truth, feature_names, feature_scales, trigger):
    """Construct one graph (e.g. from one event)

    Produce a perfect graph only contain true edges and zero false edges.
    It means the edge_index contains only true edges and the target edge
    class is always 1.
    """

    # Construct mc segments
    segments = construct_segments_mc(hits, truth)

    # Prepare the graph tuple with selected and scaled hit features
    graph = prepare_graph_matrices(hits, segments, feature_names, feature_scales, trigger)

    return graph


def prepare_graph_matrices(hits, segments, feature_names, feature_scales, trigger):

    # Prepare the graph matrices
    n_hits = hits.shape[0]
    n_edges = segments.shape[0]
    X = (hits[feature_names].values / feature_scales).astype(np.float32)
    P = (hits[['px', 'py', 'pz', 'q', 'nhits', 'particle_id']].values).astype(np.float32)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    y = np.zeros(n_edges, dtype=np.float32)
    hitId = hits['hit_id']
    trig = trigger.to_numpy().astype(np.float32)

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
    return Graph(X, Ri, Ro, y, P, trig), hitId


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
    """
    Constructs array of layer pairings.

    The layers are enumerates starting at 0. The last layer has id n_det_layers-1.
    Pairings can be done between same layer and adjecant layers. It can be configured
    by segment_type:

    'out': outgoing, layer_id difference is +1
    'inout'; outgoing and ingoing, layer_id difference si  +/-1
    'all': includes same layer pairs,  layer_id difference is  +/-1 or 0
    """

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


def form_skip_layer_pairs(first_layer_id, last_layer_id, skip, segment_type):
    """
    Constructs array of layer pairings with option to skip layers.

    The layer_id enumerates layers starting at 0. The first_layer_id and last_layer_id
    are the innermost and outermost layers that should be included in parings.

    The argument skip is a nonzero integer that defines how many layers are skipped in
    a layer pair. The first pair is (first_layer_id, first_layer_id+skip)

    Segment types are 'out' for outgoing pairs and 'all' for both
    outgoing and ingoing pairs.
    """

    # Define adjacent layers
    layerIDs = np.arange(first_layer_id, last_layer_id + 1)

    if segment_type == 'out':
        # Create only outgoing segments
        layer_pairs = np.stack([layerIDs[:-skip:1], layerIDs[skip::1]], axis=1)
    else:
        # Create ingoing and outgoing segments
        l_1 = np.concatenate((layerIDs[:-skip:1], layerIDs[skip::1]), axis=None)
        l_2 = np.concatenate((layerIDs[skip::1], layerIDs[:-skip:1]), axis=None)
        layer_pairs = np.stack((l_1, l_2), axis=-1)

    return layer_pairs
