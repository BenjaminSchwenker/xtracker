#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


"""
This file contains some common helper code for the analysis notebooks.
"""

import os
import yaml
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.data import Subset, DataLoader


def get_dataset(config):
    from xtracker.datasets.hitgraphse import HitGraphDataset
    return xtracker.datasets.hitgraphs.HitGraphDataset(get_input_dir(config))


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


def draw_sample(hits, edges, preds, labels, cut=0.5, figsize=(9, 6), rlim=(
        0, 15), zlim=(-40, 40), philim=(-np.pi, np.pi), mconly=False, fullonly=False):
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

        zp = np.array([z[edges[0, j]], z[edges[1, j]]])
        rp = np.array([r[edges[0, j]], r[edges[1, j]]])
        phip = np.array([phi[edges[0, j]], phi[edges[1, j]]])

        # Only draw true hit hitgraph
        if mconly:
            if labels[j] == 1:
                ax0.plot(zp, rp, '--', c='g')
                ax1.plot(phip, rp, '--', c='g')
            continue

        # Only draw true hit hitgraph
        if fullonly:
            ax0.plot(zp, rp, '--', c='k')
            ax1.plot(phip, rp, '--', c='k')
            continue

        # False negatives
        if (preds[j] < cut and labels[j] > cut):
            ax0.plot(zp, rp, '--', c='b')
            ax1.plot(phip, rp, '--', c='b')

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot(zp, rp, '--', c='r', alpha=preds[j])
            ax1.plot(phip, rp, '--', c='r', alpha=preds[j])

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot(zp, rp, '-', c='k', alpha=preds[j])
            ax1.plot(phip, rp, '-', c='k', alpha=preds[j])

    # Adjust axes
    ax0.set_xlabel('$z$')
    ax1.set_xlabel('$\\phi$')
    ax0.set_ylabel('$r$')
    ax1.set_ylabel('$r$')
    plt.tight_layout()


def draw_sample_xy(hits, edges, preds, labels, cut=0.5, figsize=(9, 9), mconly=False, fullonly=False):
    x = hits[:, 0] * np.cos(hits[:, 1])
    y = hits[:, 0] * np.sin(hits[:, 1])
    fig, ax0 = plt.subplots(figsize=figsize)

    # Make sure edges are integer
    edges = edges.astype(int)

    # Draw the hits
    ax0.scatter(x, y, s=2, c='k')

    # Draw the segments
    for j in range(labels.shape[0]):

        xp = np.array([x[edges[0, j]], x[edges[1, j]]])
        yp = np.array([y[edges[0, j]], y[edges[1, j]]])

        # Only draw true hit hitgraph
        if mconly:
            if labels[j] == 1:
                ax0.plot(xp, yp, '--', c='g')
            continue

        # Only draw true hit hitgraph
        if fullonly:
            ax0.plot(xp, yp, '--', c='g')
            continue

        # False negatives
        if preds[j] < cut and labels[j] > cut:
            ax0.plot(xp, yp, '--', c='b')

        # False positives
        if preds[j] > cut and labels[j] < cut:
            ax0.plot(xp, yp, '--', c='r', alpha=preds[j])

        # True positives
        if preds[j] > cut and labels[j] > cut:
            ax0.plot(xp, yp, '-', c='k', alpha=preds[j])

    return fig, ax0
