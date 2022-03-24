# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

import logging
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import pandas as pd
from tqdm import tqdm

from xtracker.gnn_tracking.Arena import Arena
from xtracker.gnn_tracking.MCTS import MCTS
from xtracker.gnn_tracking.SeqTracker import SeqTracker
from xtracker.gnn_tracking.ImTracker import ImTracker
from xtracker.gnn_tracking.TrackingSolver import TrackingSolver


class Coach():
    """
    This class executes the learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__()  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.tracker = ImTracker(self.game, self.nnet, self.args.model)

        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        self.summaries = None
        self.summary_file = None

        self.args.training.checkpoint = os.path.expandvars(self.args.training.checkpoint)

    def executeEpisode(self):
        """
        This function plays one episode.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (board, currPlayer, pi,v)
                           pi is the policy vector and v is the game score.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        solver = TrackingSolver(self.game)
        # Todo get rid of curPlayer
        self.curPlayer = 1
        episodeStep = 0

        while True:

            episodeStep += 1

            # Compute policy for next step
            if self.args.training.supervised_training:
                # Imitation learning
                pi, _ = solver.predict(board)
            else:
                # Alpha zero learning
                temp = int(episodeStep < self.args.training.tempThreshold)
                pi = self.mcts.getActionProb(board, temp=temp)

            # Sample the training example
            if episodeStep == 1:
                if np.random.rand() < self.args.training.pre_scale_examples:
                    trainExamples.append([board, self.curPlayer, pi, None])
            else:
                trainExamples.append([board, self.curPlayer, pi, None])

            # Draw the number of steps (1,2,3 ..) until we sample
            # new step into trainExamples
            delta = np.random.geometric(p=self.args.training.pre_scale_examples)

            # Compute the number of steps left in a perfect game
            steps_left = solver.steps_left(pi)

            if delta >= steps_left:
                # Decide if we want so sample last step before finishing the game
                if np.random.rand() < self.args.training.pre_stop_example:
                    board, self.curPlayer = self.game.getPerfectState(board, stop=False)
                    pi, _ = solver.predict(board)
                    trainExamples.append([board, self.curPlayer, pi, None])
                    print('Sample last action')

                # Move into perfect state and stop game
                board, self.curPlayer = self.game.getPerfectState(board, stop=True)

            else:
                # In a perfect game, the next delta steps will not
                # stop.
                action = np.random.choice(len(pi), delta, replace=False, p=pi)
                action = board.get_legal_moves()[action]
                board, self.curPlayer = self.game.getNextStateNoStop(board, self.curPlayer, action)

            if self.game.getGameEnded(board, self.curPlayer):
                r = self.game.getGameScore(board)
                return [(x[0], x[2], r) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.training.numIters + 1):
            # bookkeeping
            logging.info(f'Starting Iter #{i} ...')

            # prepare summary information
            summary = dict()
            sum_examples = 0

            # start with self play to create examples
            start_time = time.time()
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.training.maxlenOfQueue)

                for _ in tqdm(range(self.args.training.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args.model)  # reset search tree
                    episodeExamples = self.executeEpisode()
                    sum_examples += len(episodeExamples)
                    iterationTrainExamples += episodeExamples

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.training.numItersForTrainExamplesHistory:
                logging.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            summary['play_time'] = time.time() - start_time

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            # ToDo: train examples very big, make saving optional and turn off as default
            # self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # save current state of new network
            self.nnet.save_checkpoint(folder=self.args.training.checkpoint, filename='temp.pth.tar')
            # set the prev network to state of new network
            self.pnet.load_checkpoint(folder=self.args.training.checkpoint, filename='temp.pth.tar')

            # training of the network
            start_time = time.time()
            self.nnet.train(trainExamples,
                            lr=self.args.optimizer.lr,
                            epochs=self.args.optimizer.epochs,
                            batch_size=self.args.optimizer.batch_size)
            summary['train_time'] = time.time() - start_time

            logging.info('PITTING AGAINST PREVIOUS VERSION')
            start_time = time.time()
            ptracker = ImTracker(self.game, self.pnet, self.args.model)
            ntracker = ImTracker(self.game, self.nnet, self.args.model)
            arena = Arena(ptracker, ntracker, self.game)
            scores_p, scores_n = arena.playGames(self.args.training.arenaCompare)
            p_score = np.mean(scores_p)
            n_score = np.mean(scores_n)
            summary['pit_time'] = time.time() - start_time

            logging.info('NEW/PREV SCORE : %f / %f' % (n_score, p_score))
            if False:
                logging.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.training.checkpoint, filename='temp.pth.tar')
            else:
                logging.info('ACCEPTING NEW MODEL')

                # summarize the iteration
                summary['iter'] = i
                summary['train_sampled_examples'] = sum_examples
                summary['pit_nnet_score'] = n_score
                summary['pit_pnet_score'] = p_score

                # save summary
                best_iter, best_score = self.save_summary(summary)

                # save checkpoint
                self.nnet.save_checkpoint(folder=self.args.training.checkpoint, filename=self.getCheckpointFile(i))

                # save checkpoint for current best network (not necessarily the last)
                logging.info(f'Checkpoint from Iter {best_iter} is best, score is {best_score}')
                self.pnet.load_checkpoint(folder=self.args.training.checkpoint, filename=self.getCheckpointFile(best_iter))
                self.pnet.save_checkpoint(folder=self.args.training.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.training.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.training.load_folder_file[0], self.args.training.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            logging.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            logging.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            logging.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def save_summary(self, summary):
        """Save summary information.
        This implementation is currently inefficient for simplicity:
            - we build a new DataFrame each time
            - we write the whole summary file each time

        Returns:
            best_iter: iteration with best new network score in summary
            best_score: best new network score in summary
        """
        if self.summaries is None:
            self.summaries = pd.DataFrame([summary])
        else:
            self.summaries = self.summaries.append([summary], ignore_index=True)
        if self.args.training.checkpoint is not None:
            summary_file = os.path.join(self.args.training.checkpoint, 'summaries.csv')
            self.summaries.to_csv(summary_file, index=False)

        # get iter and score of best iteration
        best_idx = self.summaries.pit_nnet_score.idxmax()
        best_score = self.summaries.loc[[best_idx]].pit_nnet_score.item()
        best_iter = int(self.summaries.loc[[best_idx]].iter.item())

        return best_iter, best_score
