# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    single-player and turn-based.
    """

    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the initial board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getNextState(self, board, action):
        """
        Input:
            board: current board
            action: action taken

        Returns:
            nextBoard: board after applying action
        """
        pass

    def getValidMoves(self, board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a vector containing the indices for valid moves from
                        the current board.
        """
        pass

    def getGameEnded(self, board):
        """
        Input:
            board: current board

        Returns:
            r: False if game has not ended, otherwise True
        """
        pass

    def getGameScore(self, board):
        """
        Input:
            board: current board

        Returns:
            v: Game score
        """
        pass

    def getCanonicalForm(self, board):
        """
        Input:
            board: current board

        Returns:
            canonicalBoard: returns canonical form of board.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
