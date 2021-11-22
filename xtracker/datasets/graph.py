##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

"""
This module contains code for interacting with hit graphs.
A Graph is a namedtuple of matrices X, Ri, Ro, y, P
"""

from collections import namedtuple

import numpy as np

# A Graph is a namedtuple of matrices (X, Ri, Ro, y, P)
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y', 'P'])


def get_batch(f):
    x, y, p = f['X'], f['y'], f['P']
    Ri_rows, Ri_cols = f['Ri_rows'], f['Ri_cols']
    Ro_rows, Ro_cols = f['Ro_rows'], f['Ro_cols']
    n_edges = Ri_cols.shape[0]
    edge_index = np.zeros((2, n_edges), dtype=int)
    edge_index[0, Ro_cols] = Ro_rows
    edge_index[1, Ri_cols] = Ri_rows
    return x, edge_index, y, p


def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    return dict(X=graph.X, y=graph.y,
                Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols, P=graph.P)


def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, P, dtype=np.uint8):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y, P)


def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))


def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)


def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))


def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]
