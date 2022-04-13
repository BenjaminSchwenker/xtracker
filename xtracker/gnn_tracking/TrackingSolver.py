# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

import numpy as np
from xtracker.gnn_tracking.TrackingLogic import Board


class TrackingSolver():
    """
    This class solves the tracking game using the
    underlying mc truth event.

    It is used to create targets for training. It is also
    used for the mc truth tracking pipeline for debugging.
    """

    def __init__(self, game):
        self.game = game

    def steps_left(self, pi):
        """
        Returns number of steps left using a solver policy pi.

        The probability for stopping pi[0] is only positive, if
        all fakes are removed from the graph.

        If there are fakes left, we count their number and add 1
        for playing stop.
        """
        if pi[0] > 0:
            return 1
        else:
            return (pi > 0).sum() + 1

    def predict(self, board):
        """
        Returns predicted policy array and score.
        Note that policy array has one item more than segments on the board. This item represents the stop move.
        """

        legal_moves = board.get_legal_moves()

        # Mask active edges (y_pred==1) that are wrong (y==0)
        fakes = ((board.y == 0) & (board.y_pred == 1)).nonzero()[0]

        # Output policy array
        pi = np.zeros(shape=legal_moves.shape, dtype=np.float32)

        if fakes.shape[0] == 0:
            v = self.game.getGameScore(board)
            pi[0] = 1
        else:
            b = Board()
            b.edge_index = board.edge_index
            b.x = board.x
            b.y = board.y
            b.y_pred = np.copy(board.y_pred)
            b.y_pred[fakes] = 0

            # This is currently the main bottleneck for speed.
            # v = self.game.getGameScore(b)
            v = 1.0

            y_prime = np.copy(board.y)
            y_prime = y_prime[board.y_pred == 1]

            pi_prime = np.zeros_like(y_prime)
            pi_prime[y_prime == 0] = 1

            pi = np.insert(pi_prime, 0, 0).astype(np.float32)
            pi = pi / pi.sum()

        return pi, np.float32(v), board.trig

    def process(self, board):
        """Returns array with segment predictions and score"""
        preds = np.copy(board.y)
        return preds, 1.0, board.trig
