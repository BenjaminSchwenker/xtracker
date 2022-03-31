# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


import numpy as np


class Board():
    """
    Board class for the TrackingGame.

    The board represents a hitgraph composed
    of n_nodes and n_egdes. Each node represents
    a hit in an event and each node a candidate
    segment.

    The board is a tuple (x,egde_index,y, y_pred, next_player):
    edge_index:  2 x n_segements array with sender/receiver nodes
    x: n_nodes x n_features array with node features
    y: n_segments array with MC truth labels (True/False)
    y_pred: n_segments array with predicted labels (True/False)
    trig: mc target for trigger channel (0: bg only, 1: bg + bbbar)
    next_player: either 1 meaning game continues or 0 meaning game ends

    """

    def __init__(self, next_player=1):
        "Set up empty board configuration."
        self.next_player = next_player
        self.edge_index = None
        self.x = None
        self.y = None
        self.y_pred = None
        self.trig = None

    def get_legal_moves(self, color=None):
        """
        Returns integer array of all legal moves.
        Legal moves are either pruning an active
        edge or ending the game. For pruning an edge,
        you need to specify the index of that edge
        in the initial board. For ending the game,
        the integer -1 is used.
        @param color not used and came from previous version.
        """
        active_edges = np.where(self.y_pred == 1)[0]
        return np.insert(active_edges, 0, -1)

    def has_legal_moves(self):
        # Always possible to end the game
        return True

    def execute_move(self, move, color=None):
        """Perform the given move on the board.
           @param color not used and came from previous version.
        """

        if move < 0:
            # Move ends the game episode
            self.next_player = 0
            return 0
        else:
            # Moves prunes a candidate segment
            self.y_pred[move] = 0
            return 1

    def execute_move_nostop(self, move, color=None):
        """Perform the given move on the board.
           @param color not used and came from previous version.
        """

        # Move prunes a candidate segment
        self.y_pred[move] = 0
        return 1

    def execute_stop(self):
        """Perform stop move move on the board.
        """
        self.next_player = 0
        return 0
