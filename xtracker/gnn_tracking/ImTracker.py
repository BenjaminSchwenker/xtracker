##########################################################################
# xtracker                                                               #
# Author: Benjamin Schwenker                                             #
#                                                                        #
# See git log for contributors and copyright holders.                    #
# This file is licensed under LGPL-3.0, see LICENSE.md.                  #
##########################################################################


import numpy as np

import torch
from torch_scatter import scatter_add
from torch_scatter import scatter_max
from torch_scatter import scatter_min


class ImTracker():
    """
    This class handles the tracking of an input event using the
    given tracking net.

    This implementation batches trackign actions of sequential
    to boost speed. The algorithm has three main steps

    1) Seeding and initialization:

        A full forward pass of tracking net is executed yielding an initial policy,
        edge weights and value. All edges are removed with a policy probability
        greater than the stop probability. The remaining active edges in the resulting
        graph are 'found' or 'locked' and define the initial seed tracks.

    2) Main tracking loop:

        After seeding and initialization, the main tracking loop is entered. The
        tracking loop is N cycles and contains the follwoing steps:

        2.1) Cleaning 'found' edges:

                Cleanup 'found' or 'locked' edges from the preceeding passes that are
                competing with each other. In case of competition, remove (randomized)
                the edge with weakest edge weight e.

                -> hope that can be done with scatter_min operation

        2.2) Reattaching edges:

                Add (reattach) all edges from the initial graph that do not
                compete with 'found' or 'locked' edges after cleaning. This
                creates a new candidate graph.

                -> hope that can be done with scatter_max operation

        2.3) Execute forward passes:

                Execute a forward pass of the tracking net on the reattached graph
                to optain a new policy vector and edge weights. Prune all edges with
                policy probability higher than stop probability, i.e. unroll the policy.


                Instead of unrolling the complete policy, we can introduce NI half steps
                steps that remove the more likely upper half of edges from the graph and
                execute another forward pass of the tracking net afterwards. This must be
                followed by an final pass the unrolls the policy yielding a new set of
                'found' or 'locked edges'.

    3) Postprocessing:

        Read connected components from graph forming the final track sets. Final attempt to
        clean / improve tracks. Compute quality indicators and kinemetic variables (vertices
        or track parameters) from node embeddings.

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
        e = None

        # Play game
        while it < self.args.nIter:
            it += 1
            if self.args.verbose:
                print("Turn ", str(it), "Score ", str(self.game.getGameScore(board)))

            if it > 1:

                self.reattach_edges(board, e)

                # in between passes
                for istep in range(self.args.mid_steps):
                    active_edges = np.where(board.y_pred == 1)[0]
                    legal_moves = np.insert(active_edges, 0, -1)
                    p, v, e_new = self.nnet.predict(board)

                    if self.args.update_e:
                        e[board.y_pred.astype(bool)] = e_new

                    actions = legal_moves[p > p[0]]

                    num_actions = int(actions.shape[0] / 2)
                    # num_actions =  int(actions.shape[0]/(max_steps+1-istep))

                    if num_actions == 0:
                        break

                    index_array = p.argsort()[::-1][:num_actions]
                    legal_moves[index_array]
                    board.y_pred[legal_moves[index_array]] = 0

            # Final pass
            active_edges = np.where(board.y_pred == 1)[0]
            legal_moves = np.insert(active_edges, 0, -1)

            if it == 1:
                p, v, e = self.nnet.predict(board)
            else:
                p, v, e_new = self.nnet.predict(board)
                if self.args.update_e:
                    e[board.y_pred.astype(bool)] = e_new

            # Select all actions with higher likelyhood than stop
            actions = legal_moves[p > p[0]]

            # Execute all actions: remove marked segments
            board.y_pred[actions] = 0

            # Stop the game
            board.next_player = 0

        if self.args.verbose:
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameScore(board)))

        return board.y_pred, self.game.getGameScore(board)

    def reattach_edges(self, board, e=None):
        """Reattach active egdes to board.
        """

        # Create list of good edges found so far
        good_edges = (board.y_pred > 0).nonzero()[0]

        # Reattach all edges from initial board
        board.y_pred = np.ones_like(board.y)

        # Remove all edges from the board that compete with a good edge
        start, end = board.edge_index
        sender, receiver = board.edge_index[:, good_edges]
        board.y_pred = (np.isin(start, sender, invert=True) & np.isin(end, receiver, invert=True))

        # Keep track of indices after filterings
        num_nodes = board.x.shape[0]
        num_edges = board.y_pred.shape[0]

        edge_index = board.edge_index[:, board.y_pred.astype(bool)]
        e_temp = e[board.y_pred.astype(bool)]

        # Add noise to e
        noise = np.random.normal(0, self.args.noise_sigma, e_temp.shape[0])
        e_temp += noise

        # Select nodes for proposing candidate edges
        in_degree, out_degree = get_degrees(num_nodes, edge_index)
        in_open = (in_degree > 0)
        out_open = (out_degree > 0)

        # Select edges that maximize e
        idx_helper = np.arange(num_edges)
        idx_helper = idx_helper[board.y_pred.astype(bool)]

        argmax_in, argmax_out = max_edges(num_nodes, edge_index, e_temp)
        argmax_in = argmax_in[in_open]
        argmax_out = argmax_out[out_open]

        # Clear board for adding the new candidates
        board.y_pred = np.zeros_like(board.y)

        # Attach new proposed candidates
        board.y_pred[idx_helper[argmax_in]] = 1
        board.y_pred[idx_helper[argmax_out]] = 1

        # Attach all found edges
        board.y_pred[good_edges] = 1

        # Reopen the game
        board.next_player = 1


def get_degrees(num_nodes, edge_index):
    start, end = torch.LongTensor(edge_index)
    ones = torch.ones(num_nodes)

    in_degree = scatter_add(ones[start], end, dim=0, dim_size=num_nodes)
    out_degree = scatter_add(ones[end], start, dim=0, dim_size=num_nodes)
    return in_degree.data.cpu().numpy(), out_degree.data.cpu().numpy()


def max_edges(num_nodes, edge_index, e):
    start, end = torch.LongTensor(edge_index)
    e = torch.FloatTensor(e)
    ones = torch.ones((num_nodes, 1))

    max_in, argmax_in = scatter_max(e[:, None] * ones[start], end, dim=0, dim_size=num_nodes)
    max_out, argmax_out = scatter_max(e[:, None] * ones[end], start, dim=0, dim_size=num_nodes)
    return argmax_in.squeeze(1).data.cpu().numpy(), argmax_out.squeeze(1).data.cpu().numpy()


def min_edges(num_nodes, edge_index, e):
    start, end = torch.LongTensor(edge_index)
    e = torch.FloatTensor(e)
    ones = torch.ones((num_nodes, 1))

    min_in, argmin_in = scatter_min(e[:, None] * ones[start], end, dim=0, dim_size=num_nodes)
    min_out, argmin_out = scatter_min(e[:, None] * ones[end], start, dim=0, dim_size=num_nodes)
    return argmin_in.squeeze(1).data.cpu().numpy(), argmin_out.squeeze(1).data.cpu().numpy()
