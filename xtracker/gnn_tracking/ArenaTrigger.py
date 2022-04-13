# xtracker (Neural network based trackfinding for Belle II)
# Author: The xtracker developers
#
# See git log for contributors and copyright holders.
# This file is licensed under GPLv3+ licence, see LICENSE.md.

from tqdm import tqdm


class ArenaTrigger():
    """
    An Arena class where two players competes for the best game score.
    """

    def __init__(self, player1, player2, game, tracker):
        """
        Input:
            player 1,2: two functions that takes board as input and return score and pred
            game: Game object
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.tracker = tracker

    def playGame(self):
        """
        Executes one episode of a game.

        Returns:
            score at game end
        """
        board = self.game.getInitBoard(training=False)
        x = self.tracker.embed_hits(board)
        pred_trig = self.player1.predict(x)[0, 0]
        true_trig = board.trig.numpy()[0, 0]
        return self.getGameScore(pred_trig, true_trig)

    def playGames(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            scores_1: list of scores for fist num/2 plays for player_1
            scores_2: list of scores for fist num/2 plays for player_2
        """
        num = int(num / 2)
        scores_1 = []
        scores_2 = []

        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame()
            scores_1.append(gameResult)

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame()
            scores_2.append(gameResult)

        return scores_1, scores_2

    def getGameScore(self, pred_trig, truth_trig):
        """
        Returns score for trigger game
        """

        if truth_trig == round(pred_trig):
            return 1.0
        else:
            return 0.0
