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
from .TrackingNNet import TriggerNNetwork

use_cuda = torch.cuda.is_available()


class NNetWrapper(NeuralNet):
    def __init__(self,embedding_dim,layer_size,n_update_iters):
        self.nnet = tnet(embedding_dim, layer_size, n_graph_iters=n_update_iters)

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
                l_trig = self.loss_trig(target_trigs, out_trig)
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

        self.nnet.eval()
        with torch.no_grad():
            pi, v, e, trig = self.nnet(x, edge_index)

        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()[0], e.data.cpu().numpy(), trig.data.cpu().numpy()

    def predict_event_embedding(self, board):
        """
        board: np array with board
        """

        # preparing input
        edge_index = torch.LongTensor(board.edge_index[:, board.y_pred.astype(bool)])
        x = torch.FloatTensor(board.x)

        self.nnet.eval()
        with torch.no_grad():
            xs = self.nnet.forward_embed(x, edge_index)

        return xs.data.cpu().numpy()

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
            import errno
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
        map_location = None if use_cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


class NNetWrapperTrigger(NeuralNet):
    def __init__(self,embedding_dim,layer_size):
        self.nnet = TriggerNNetwork(embedding_dim, layer_size)

        if use_cuda:
            self.nnet.cuda()

    def train(self, examples, lr, epochs, batch_size):
        """
        examples: list of examples, each example is of form (x, y)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=lr)

        loss_fn = torch.nn.BCELoss()

        for epoch in range(epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            trig_losses = AverageMeter()

            batch_count = int(len(examples) / batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=batch_size)
                x, y = list(zip(*[examples[i] for i in sample_ids]))

                x = torch.FloatTensor(x[0])
                y = torch.FloatTensor(y[0])

                # predict
                if use_cuda:
                    x = x.contiguous().cuda()
                    y = y.contiguous().cuda()

                # compute output
                output = self.nnet(x)

                # calculate loss
                loss = loss_fn(output, y.reshape(-1, 1))

                # record loss
                trig_losses.update(loss.item())
                t.set_postfix(Loss_trig=trig_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, x):
        """
        x: np array
        """

        # preparing input
        x = torch.FloatTensor(x)

        self.nnet.eval()
        with torch.no_grad():
            trig = self.nnet(x)

        return trig.data.cpu().numpy()

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
            import errno
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
        map_location = None if use_cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
