# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


from .Game import Game
from .TrackingLogic import Board


import numpy as np
from itertools import cycle


class TrackingGame(Game):
    """
    Class implementation for the TrackingGame.

    Author: Benjamin Schwenker, github.com/BenjaminSchwenker
    Date: Oct 14, 2020.
    """

    def __init__(self, train_data_loader=None, valid_data_loader=None):
        if train_data_loader is None or valid_data_loader is None:
            self.train_data_loader = None
            self.valid_data_loader = None
        else:
            self.train_data_loader = cycle(train_data_loader)
            self.valid_data_loader = cycle(valid_data_loader)

    def getInitBoardFromBatch(self, batch):
        """
        Returns:
            startBoard: a representation of the initial board
        """
        x, edge_index, y, p, trig = batch

        b = Board()
        b.edge_index = edge_index.numpy().copy()
        b.x = x.numpy().copy()
        b.y = y.numpy().copy()
        b.y_pred = np.ones_like(b.y)
        b.trig = trig.numpy()
        b.trig_pred = np.ones_like(b.trig)

        return b

    def getInitBoard(self, training=True):
        """
        Returns:
            startBoard: a representation of the initial board
        """
        if training:
            batch = next(self.train_data_loader)
        else:
            batch = next(self.valid_data_loader)

        x, edge_index, y, p, trig = batch

        b = Board()
        b.edge_index = edge_index.numpy().copy()
        b.x = x.numpy().copy()
        b.y = y.numpy().copy()
        b.y_pred = np.ones_like(b.y)
        b.trig = trig
        b.trig_pred = np.ones_like(b.trig)

        return b

    def getPerfectState(self, board, stop=True):
        """
        Input:
            board: current board
            action: action taken (must be valid)

        Returns:
            nextBoard: board after applying action
        """
        b = Board()
        b.edge_index = board.edge_index
        b.x = board.x
        b.y = board.y
        b.y_pred = np.copy(board.y)
        b.trig = board.trig
        b.trig_pred = np.copy(board.trig)
        next_player = 1
        if stop:
            next_player = b.execute_stop()
        return (b, next_player)

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            action: action taken (must be valid)

        Returns:
            nextBoard: board after applying action
        """
        b = Board()
        b.edge_index = board.edge_index
        b.x = board.x
        b.y = board.y
        b.y_pred = np.copy(board.y_pred)
        b.trig = board.trig
        b.trig_pred = np.copy(board.trig_pred)
        next_player = b.execute_move(action)
        return (b, next_player)

    def getNextStateNoStop(self, board, player, action):
        """
        Input:
            board: current board
            action: action taken (must be valid)

        Returns:
            nextBoard: board after applying action
        """
        b = Board()
        b.edge_index = board.edge_index
        b.x = board.x
        b.y = board.y
        b.y_pred = np.copy(board.y_pred)
        b.trig = board.trig
        b.trig_pred = np.copy(board.trig_pred)
        next_player = b.execute_move_nostop(action)
        return (b, next_player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board

        Returns:
            validMoves: a vector containing the indices for valid moves from
                        the current board.
        """
        return board.get_legal_moves()

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board

        Returns:
            r: False if game has not ended, otherwise True
        """
        if board.next_player == 1:
            return False
        else:
            return True

    def getGameScore(self, board):
        """
        Input:
            board: current board

        Returns:
            v: Game score
        """
        # FIXME: Old stuff, remove it at some point
        # return sklearn.metrics.accuracy_score(board.y, board.y_pred)

        # Here is the moment to remove all wrong edges
        true_edge_index = board.edge_index[:, board.y.astype(bool)]

        # These are the edges i finally want to keep
        edge_index = board.edge_index[:, board.y_pred.astype(bool)]

        # Compute the score
        score = 0
        nhits = board.x.shape[0]

        for ihit in range(nhits):

            # Get list of previous hits
            true_ends = (true_edge_index[1, :] == ihit).nonzero()[0]
            true_prehit = true_edge_index[0, true_ends]

            ends = (edge_index[1, :] == ihit).nonzero()[0]
            prehit = edge_index[0, ends]

            if np.array_equal(prehit, true_prehit):
                score += 0.5

            # Get list of next hits
            true_starts = (true_edge_index[0, :] == ihit).nonzero()[0]
            true_posthit = true_edge_index[1, true_starts]

            starts = (edge_index[0, :] == ihit).nonzero()[0]
            posthit = edge_index[1, starts]

            if np.array_equal(posthit, true_posthit):
                score += 0.5

        if nhits > 0:
            score /= nhits

        return score

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board

        Returns:
            canonicalBoard: returns canonical form of board.
        """
        return board

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        # FIXME: not sure if i want to add x
        return board.y_pred.tostring()  # +board.x.tostring()

    @staticmethod
    def display(board):
        print("Predicted segments: \n", board.y_pred)
        print("MC truth segments:  \n", board.y)
        print("Predicted trigger: \n", board.trig_pred)
        print("MC truth trigger:  \n", board.trig)
