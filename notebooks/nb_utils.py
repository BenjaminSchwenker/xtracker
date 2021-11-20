#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################


"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import yaml
import pickle
from collections import namedtuple

from sklearn.cluster import DBSCAN
from scipy.sparse import dok_matrix


# Externals
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.data import Subset, DataLoader


def get_dataset(config):
    from xtracker.datasets.hitgraphs_sparse import HitGraphDataset
    return xtracker.datasets.hitgraphs_sparse.HitGraphDataset(get_input_dir(config))


def get_test_data_loader(config, n_test=16, batch_size=1):
    # Take the test set from the back
    full_dataset = get_dataset(config)
    test_indices = len(full_dataset) - 1 - torch.arange(n_test)
    test_dataset = Subset(full_dataset, test_indices.tolist())
    return DataLoader(test_dataset, batch_size=batch_size,
                      collate_fn=Batch.from_data_list)


def load_summaries(output_dir):
    summary_file = os.path.join(output_dir, 'summaries.csv')
    return pd.read_csv(summary_file)


# Define our Metrics class as a namedtuple
Metrics = namedtuple('Metrics', ['accuracy', 'precision', 'recall', 'f1',
                                 'prc_precision', 'prc_recall', 'prc_thresh',
                                 'roc_fpr', 'roc_tpr', 'roc_thresh', 'roc_auc'])


def convert_to_sparse_tuple(graph):
    """Convert dense graph to sparse tuple"""
    import xtracker.datasets.graph
    f = datasets.graph.graph_to_sparse(graph)

    x, y = f['X'], f['y']
    Ri_rows, Ri_cols = f['Ri_rows'], f['Ri_cols']
    Ro_rows, Ro_cols = f['Ro_rows'], f['Ro_cols']
    n_edges = Ri_cols.shape[0]
    edge_index = np.zeros((2, n_edges), dtype=int)
    edge_index[0, Ro_cols] = Ro_rows
    edge_index[1, Ri_cols] = Ri_rows
    return x, edge_index, y


def compute_metrics(preds, targets, threshold=0.5):
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    # Decision boundary metrics
    y_pred, y_true = (preds > threshold), (targets > threshold)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    # Precision recall curves
    prc_precision, prc_recall, prc_thresh = sklearn.metrics.precision_recall_curve(y_true, preds)
    # ROC curve
    roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, preds)
    roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)
    # Organize metrics into a namedtuple
    return Metrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                   prc_precision=prc_precision, prc_recall=prc_recall, prc_thresh=prc_thresh,
                   roc_fpr=roc_fpr, roc_tpr=roc_tpr, roc_thresh=roc_thresh, roc_auc=roc_auc)


def plot_train_history(summaries, figsize=(12, 10), loss_yscale='linear'):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axs = axs.flatten()

    # Plot losses
    axs[0].plot(summaries.epoch, summaries.train_loss, label='Train')
    axs[0].plot(summaries.epoch, summaries.valid_loss, label='Validation')
    axs[0].set_yscale(loss_yscale)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc=0)

    # Plot accuracies
    axs[1].plot(summaries.epoch, summaries.valid_acc, label='Validation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_ylim(bottom=0, top=1)
    axs[1].legend(loc=0)

    # Plot model weight norm
    axs[2].plot(summaries.epoch, summaries.l2)
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Model L2 weight norm')

    # Plot learning rate
    axs[3].plot(summaries.epoch, summaries.lr)
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Learning rate')

    plt.tight_layout()


def plot_metrics(preds, targets, metrics):
    # Prepare the values
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    labels = targets > 0.5

    # Create the figure
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 5))

    # Plot the model outputs
    binning = dict(bins=25, range=(0, 1), histtype='step', log=True)
    ax0.hist(preds[labels == False], label='fake', **binning)
    ax0.hist(preds[labels], label='real', **binning)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot precision and recall
    ax1.plot(metrics.prc_thresh, metrics.prc_precision[:-1], label='purity')
    ax1.plot(metrics.prc_thresh, metrics.prc_recall[:-1], label='efficiency')
    ax1.set_xlabel('Model threshold')
    ax1.legend(loc=0)

    # Plot the ROC curve
    ax2.plot(metrics.roc_fpr, metrics.roc_tpr)
    ax2.plot([0, 1], [0, 1], '--')
    ax2.set_xlabel('False positive rate')
    ax2.set_ylabel('True positive rate')
    ax2.set_title('ROC curve, AUC = %.3f' % metrics.roc_auc)

    plt.tight_layout()


def plot_outputs_roc(preds, targets, metrics):
    # Prepare the values
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    labels = targets > 0.5

    # Create the figure
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 5))

    # Plot the model outputs
    binning = dict(bins=25, range=(0, 1), histtype='step', log=True)
    ax0.hist(preds[labels == False], label='fake', **binning)
    ax0.hist(preds[labels], label='real', **binning)
    ax0.set_xlabel('Model output')
    ax0.legend(loc=0)

    # Plot the ROC curve
    ax1.plot(metrics.roc_fpr, metrics.roc_tpr)
    ax1.plot([0, 1], [0, 1], '--')
    ax1.set_xlabel('False positive rate')
    ax1.set_ylabel('True positive rate')
    ax1.set_title('ROC curve, AUC = %.3f' % metrics.roc_auc)
    plt.tight_layout()


def draw_sample(hits, edges, preds, labels, cut=0.5, figsize=(15, 7), rlim=(
        0, 1.4), zlim=(-1, 1), philim=(-1.2, 1.2), mconly=False, fullonly=False):
    r = hits[:, 0]
    phi = hits[:, 1]
    z = hits[:, 2]

    # Make sure edges are integer
    edges = edges.astype(int)

    # Prepare the figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    # Draw the hits (r, phi, z)
    ax0.scatter(z, r, s=2, c='k')
    ax1.scatter(phi, r, s=2, c='k')

    ax0.set(xlim=zlim, ylim=rlim)
    ax1.set(xlim=philim, ylim=rlim)

    # Draw the segments
    for j in range(labels.shape[0]):

        # Only draw true hit hitgraph
        if mconly:
            if labels[j] == 1:
                ax0.plot([z[edges[0, j]], z[edges[1, j]]],
                         [r[edges[0, j]], r[edges[1, j]]],
                         '--', c='g')
                ax1.plot([phi[edges[0, j]], phi[edges[1, j]]],
                         [r[edges[0, j]], r[edges[1, j]]],
                         '--', c='g')
            continue

        # Only draw true hit hitgraph
        if fullonly:
            ax0.plot([z[edges[0, j]], z[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '--', c='k')
            ax1.plot([phi[edges[0, j]], phi[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '--', c='k')
            continue

        # False negatives
        if (preds[j] < cut and labels[j] > cut):
            ax0.plot([z[edges[0, j]], z[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '--', c='b')
            ax1.plot([phi[edges[0, j]], phi[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '--', c='b')

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot([z[edges[0, j]], z[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '--', c='r', alpha=preds[j])
            ax1.plot([phi[edges[0, j]], phi[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '--', c='r', alpha=preds[j])

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot([z[edges[0, j]], z[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '-', c='k', alpha=preds[j])
            ax1.plot([phi[edges[0, j]], phi[edges[1, j]]],
                     [r[edges[0, j]], r[edges[1, j]]],
                     '-', c='k', alpha=preds[j])

    # Adjust axes
    ax0.set_xlabel('$z$')
    ax1.set_xlabel('$\\phi$')
    ax0.set_ylabel('$r$')
    ax1.set_ylabel('$r$')
    plt.tight_layout()


def draw_sample_xy(hits, edges, preds, labels, cut=0.5, figsize=(16, 16), mconly=False, fullonly=False):
    x = hits[:, 0] * np.cos(hits[:, 1])
    y = hits[:, 0] * np.sin(hits[:, 1])
    fig, ax0 = plt.subplots(figsize=figsize)

    # Make sure edges are integer
    edges = edges.astype(int)

    # Draw the hits
    ax0.scatter(x, y, s=2, c='k')

    # Draw the segments
    for j in range(labels.shape[0]):

        # Only draw true hit hitgraph
        if mconly:
            if labels[j] == 1:
                ax0.plot([x[edges[0, j]], x[edges[1, j]]],
                         [y[edges[0, j]], y[edges[1, j]]],
                         '--', c='g')

            continue

        # Only draw true hit hitgraph
        if fullonly:
            ax0.plot([x[edges[0, j]], x[edges[1, j]]],
                     [y[edges[0, j]], y[edges[1, j]]],
                     '--', c='g')

            continue

        # False negatives
        if preds[j] < cut and labels[j] > cut:
            ax0.plot([x[edges[0, j]], x[edges[1, j]]],
                     [y[edges[0, j]], y[edges[1, j]]],
                     '--', c='b')

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot([x[edges[0, j]], x[edges[1, j]]],
                     [y[edges[0, j]], y[edges[1, j]]],
                     '--', c='r', alpha=preds[j])

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot([x[edges[0, j]], x[edges[1, j]]],
                     [y[edges[0, j]], y[edges[1, j]]],
                     '-', c='k', alpha=preds[j])

    return fig, ax0


def compute_tracks_from_gnn(edges, preds, labels, cut=0.5, min_mc_hits=3):
    """Compute pr and mc tracks from segment classifier outputs using a cut approach

    The mc tracks are computed as connected components of the mc hit graph. The mc graph as
    connects hits only with True segments.

    The pr tracks are computed as connected components of a pr hit graph. The pr graph
    connects hits only when the edge probability exceeds a cut.

    Args:
        edges: edge index, subset of segments on hitgraph
        preds: GNN predictions on edge index
        labels: Ground truth on edge index

    Returns:
        mc_tracks: networkx.connected_components retun value
        pr_tracks: networkx.connected_components retun value
    """

    mc_G = nx.Graph()
    pr_G = nx.Graph()

    for j in range(labels.shape[0]):

        # Only True edges are added to mc graph
        if labels[j] == 1:
            mc_G.add_edge(edges[0, j], edges[1, j])

        # Positive edges are added to pr graph
        if preds[j] > cut:
            pr_G.add_edge(edges[0, j], edges[1, j])

    mc_tracks = sorted(nx.connected_components(mc_G), key=len, reverse=True)
    pr_tracks = sorted(nx.connected_components(pr_G), key=len, reverse=True)

    def filter_short_tracks(tracks):
        return [track for track in tracks if len(track) >= min_mc_hits]

    return filter_short_tracks(mc_tracks), filter_short_tracks(pr_tracks)


def compute_foms_from_tracks(mc_tracks, pr_tracks, min_hits=3):
    """Compute figures of merit from tracks.

    A mc track is found when at least 4 hits in it are present in at least one connected
    component in the output graph. These tracks are called matched.

    Args:
      mc_tracks: networkx.connected_components retun value
      pr_tracks: networkx.connected_components retun value

    Returns:
      finding_efficiency: A `float` fraction of found mc tracks
      hit_efficiency: A `float` fraction of found hits from pr tracks
      hit_purity: A `float` fraction of matched hits to all all hits in found tracks
    """

    finding_efficiency = 0
    hit_purity = 0
    hit_efficiency = 0
    n_matched = 0

    for i_mc, mc_track in enumerate(mc_tracks):
        for i_pr, pr_track in enumerate(pr_tracks):
            matched_hits = float(len(mc_track & pr_track))
            if matched_hits >= min_hits:
                n_matched += 1
                hit_purity += matched_hits / len(pr_track)
                hit_efficiency += matched_hits / len(mc_track)
                break

    hit_purity = hit_purity / n_matched
    hit_efficiency = hit_efficiency / n_matched
    finding_efficiency = float(n_matched) / len(mc_tracks)

    return finding_efficiency, hit_efficiency, hit_purity


def compute_tracks_from_gnn_dbscan(edges, preds, labels, n_hits, eps=0.3, min_samples=1, min_mc_hits=3):

    MC = dok_matrix((n_hits, n_hits), dtype=np.float32)
    PR = dok_matrix((n_hits, n_hits), dtype=np.float32)

    # Add segments to sparse pairwise distance
    for j in range(labels.shape[0]):
        hit_snd = edges[0, j]
        hit_rec = edges[1, j]
        pred = preds[j]
        label = labels[j]
        if label == 1:
            MC[hit_snd, hit_rec] = label * 0.5 * eps
            MC[hit_rec, hit_snd] = label * 0.5 * eps
        if pred > 0.1:
            PR[hit_snd, hit_rec] = 1 - pred
            PR[hit_rec, hit_snd] = 1 - pred

    db_pr = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(PR.tocsr())
    db_mc = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(MC.tocsr())

    def get_tracks(labels):
        tracks = {}
        for n, label in enumerate(labels):
            if label in tracks:
                tracks[label].append(n)
            else:
                tracks[label] = [n]
        return [set(track) for track in tracks.values() if len(track) >= min_mc_hits]

    return get_tracks(db_mc.labels_), get_tracks(db_pr.labels_)
