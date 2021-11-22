##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

import numpy as np


class SeqTracker():
    """
    This class handles the tracking of an input event using the
    given tracking net.

    The SeqTracker is reinforcement learning agent that can decide
    to either remove an active edge in the event or finish
    tracking on the event. After each action, a full forward pass
    of the tracking net is executed yielding a new policy vector.

    This is the reference implementation. It is rather slow on big
    dense events but yields very good performance with loaded net.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

    def process(self, board):
        """
        This function performs tracking on the input board and returns
        the final edge prediction and an event score.

        Returns:
            pred: edge predictions on board
            score: game score
        """

        it = 0

        # Play game
        while not self.game.getGameEnded(board, 1):
            it += 1

            active_edges = np.where(board.y_pred == 1)[0]
            legal_moves = np.insert(active_edges, 0, -1)

            p, v, _ = self.nnet.predict(board)

            action_idx = np.argmax(p)
            action = legal_moves[action_idx]

            if action < 0:
                board.next_player = 0
            else:
                board.y_pred[action] = 0

        if self.args.verbose:
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameScore(board)))

        return board.y_pred, self.game.getGameScore(board)
