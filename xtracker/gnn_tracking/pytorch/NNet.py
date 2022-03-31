# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.


import os
import numpy as np
from tqdm import tqdm

from xtracker.utils import AverageMeter
from xtracker.gnn_tracking.NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .TrackingNNet import TrackingNNet as tnet

use_cuda = torch.cuda.is_available()


class NNetWrapper(NeuralNet):
    def __init__(self):
        self.nnet = tnet(input_dim=3, hidden_dim=64, n_graph_iters=3,
                         hidden_activation='ReLU', layer_norm=True)

        if use_cuda:
            self.nnet.cuda()

    def train(self, examples, lr, epochs, batch_size):
        """
        examples: list of examples, each example is of form (board, pi, v, trig)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=lr)

        for epoch in range(epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            trig_losses = AverageMeter()

            batch_count = int(len(examples) / batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs, trigs = list(zip(*[examples[i] for i in sample_ids]))

                # print('benni boards ', type(boards[0]))
                # print('benni pis ', type(pis[0]))
                # print('benni vs ', type(vs[0]))
                # print('benni trigs ', type(trigs[0]))

                edge_index = torch.LongTensor(boards[0].edge_index[:, boards[0].y_pred.astype(bool)])
                x = torch.FloatTensor(boards[0].x)

                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                target_trigs = torch.FloatTensor(np.array(trigs).astype(np.float64))

                # print('target_pis ', target_pis.size())
                # print('target_vs ', target_vs.size())
                # print('target_trigs ', target_trigs.size())

                # predict
                if use_cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()
                    target_trigs = target_trigs.contiguous().cuda()

                # compute output
                out_pi, out_v, out_e, out_trig = self.nnet(x, edge_index)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                l_trig = self.loss_v(target_trigs, out_trig)
                total_loss = l_pi + l_v + l_trig

                # record loss
                pi_losses.update(l_pi.item())
                v_losses.update(l_v.item())
                trig_losses.update(l_trig.item())
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, Loss_trig=trig_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """

        # preparing input
        edge_index = torch.LongTensor(board.edge_index[:, board.y_pred.astype(bool)])
        x = torch.FloatTensor(board.x)

        # FIXME: This is the old code, kept for reference
        # board = torch.FloatTensor(board.astype(np.float64))
        # if use_cuda: board = board.contiguous().cuda()
        # board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            pi, v, e, trig = self.nnet(x, edge_index)

        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()[0], e.data.cpu().numpy(), trig.data.cpu().numpy()

    def predict_e(self, board):
        """
        board: np array with board
        """

        # preparing input
        edge_index = torch.LongTensor(board.edge_index[:, board.y_pred.astype(bool)])
        x = torch.FloatTensor(board.x)

        self.nnet.eval()
        with torch.no_grad():

            # Apply input network to get hidden representation
            x = self.nnet.input_network(x)

            # Loop over message passing layers
            for _ in range(self.nnet.n_graph_iters):
                x = self.nnet.step_gnn(x, edge_index)

            logit_e = self.nnet.edge_network(x, edge_index)
            e = torch.sigmoid(logit_e)

        return e.data.cpu().numpy()

    def predict_cached(self, board):
        """
        board: np array with board
        """

        # preparing input
        edge_index = torch.LongTensor(board.edge_index[:, board.y_pred.astype(bool)])
        x = torch.FloatTensor(board.x)

        self.nnet.eval()
        with torch.no_grad():
            xs, x_summed = self.nnet.forward_cached(x, edge_index)

        return [x.data.cpu().numpy() for x in xs], x_summed.data.cpu().numpy()

    def get_1hop_mask(self, edge_index, source):
        """
        Get edge mask for 1hop graph around source
        """
        mask = ((edge_index[0, :] == source) | (edge_index[1, :] == source))
        return mask

    def predict_embed(self, edge_index, last_xs, last_x_sum, node_mask):
        """
        board: np array with board
        """

        # preparing input
        edge_index = torch.LongTensor(edge_index)
        xs = [torch.FloatTensor(last_x) for last_x in last_xs]
        x_sum = torch.FloatTensor(last_x_sum)

        self.nnet.eval()
        with torch.no_grad():

            # Update the sum of cached node embeddings
            x_sum -= xs[-1][node_mask].sum()

            for i in range(self.nnet.n_graph_iters):
                x = xs[i]

                x_new = self.nnet.node_step(x, edge_index, node_mask)

                # Update the cached node embeddings
                xs[i + 1][node_mask] = x_new

            # Update the sum of cached node embeddings
            x_sum += x_new.sum()

        return [x.data.cpu().numpy() for x in xs], x_sum.data.cpu().numpy()

    def predict_logits(self, edge_index, x):
        """
        board: np array with board
        """

        # preparing input
        edge_index = torch.LongTensor(edge_index)
        x = torch.FloatTensor(x)

        self.nnet.eval()
        with torch.no_grad():
            e = self.nnet.prune_network(x, edge_index)

        return e.data.cpu().numpy()

    def predict_finalize_fast(self, edge_index, x, x_sum, e=None):
        """
        board: np array with board
        """

        self.nnet.eval()
        with torch.no_grad():
            if e is None:
                edge_index = torch.LongTensor(edge_index)
                x = torch.FloatTensor(x)
                pi, v, e = self.nnet.forward_finalize(x, edge_index)
            else:
                x_sum = torch.FloatTensor(x_sum)
                e = torch.FloatTensor(e)
                pi, v, e = self.nnet.forward_finalize_fast(e, x_sum)

        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()[0], e.data.cpu().numpy()

    def predict_finalize(self, board, x):
        """
        board: np array with board
        """

        # preparing input
        edge_index = torch.LongTensor(board.edge_index[:, board.y_pred.astype(bool)])
        x = torch.FloatTensor(x)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet.forward_finalize(x, edge_index)

        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_trig(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        folder = os.path.expandvars(folder)
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        filepath = os.path.expandvars(filepath)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if use_cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
