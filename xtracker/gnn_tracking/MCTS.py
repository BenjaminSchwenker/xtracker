##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################

import logging
import math

import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, training=True):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.training = training
        self.Qsa = {}   # stores Q values for s,a
        self.Nsa = {}   # stores #times edge s,a was visited
        self.Ns = {}    # stores #times board s was visited
        self.Ps = {}    # stores initial policy (returned by neural net)
        self.Vs = {}    # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        legal_moves = canonicalBoard.get_legal_moves().tolist()

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in legal_moves]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Returns:
            v: the value of the current canonicalBoard
        """
        s = self.game.stringRepresentation(canonicalBoard)

        if self.game.getGameEnded(canonicalBoard, 1):
            # terminal node
            if self.training:
                # return the true game score in training
                return self.game.getGameScore(canonicalBoard)
            else:
                # return estimated score during inference
                _, v = self.nnet.predict(canonicalBoard)
                return v

        if s not in self.Ps:
            # leaf node
            p, v = self.nnet.predict(canonicalBoard)
            d = np.random.dirichlet(p.shape[0] * [self.args.alpha])

            self.Ps[s] = self.args.x * p + (1 - self.args.x) * d
            self.Vs[s] = self.game.getValidMoves(canonicalBoard, 1)
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = None

        # pick the action with the highest upper confidence bound
        for idx, a in enumerate(valids):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][idx] * math.sqrt(self.Ns[s]) / (
                    1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][idx] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        # Selected best action
        a = best_act

        # Execute best action
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
