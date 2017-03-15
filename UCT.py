# [SublimeLinter flake8-ignore:+E261,-E222,-E231]

# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

import random
import numpy as np
import copy
from ncaa_bracket import SingleEliminationBracket, ncaa_custom_score
import ncaa_trueskill as nct
import json
import datetime


nate_silver_picks = None


def load_nate_silver(db, file):
    global nate_silver_picks
    nate_silver_picks = []
    with open(file, 'rb') as f:
        for line in f:
            if line.strip():
                team_name, seed = line.split(',')
                # Get team from database
                team = db[team_name]
                team.seed = float(seed)
                nate_silver_picks.append(team)
    return nate_silver_picks


def get_random_opponent_picks(teamlist):
    if random.random() < 0.85:
        # 85% of the time simulate against nate silver
        return nate_silver_picks
    else:
        # For the rest of the time randomly simulate the bracket
        opponent = SingleEliminationBracket(teamlist).simulate_random_bracket()
        opponent_picks = [m.winner for m in opponent]
    return opponent_picks


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic 
        zero-sum game, although they can be enhanced and made quicker, for example by using a 
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    def __init__(self, teamlist):
            self.playerJustMoved = 2 # At the root pretend the player just moved is player 2 - player 1 has the first move
            # Number of matches in single elimination
            self.nmatches = len(teamlist) - 1

            # Position index of the current match for picks
            self.match_index = 0
            # Position to insert available moves
            self.end_index = self.nmatches + 1

            # This is the initial list of valid moves.  Each pair of indices corresponds to available picks.
            # It is initialized to the initial team list (round 1) and filled in as moves are made.
            # Length is 2*nmatches.
            self.available_moves = teamlist + [None]*(self.nmatches - 1)

            # List of picks made
            self.picks = [None]*self.nmatches

            # This is a placeholder for the final simulation of the game when doing Monte Carlo evaluations
            self.simulated_game = None
            self.opponent_score = None
            self.opponent_picks = None
            self.teamlist = teamlist
    
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        return copy.deepcopy(self)

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """

        # Apply the move as a pick in the list of player picks
        self.picks[self.match_index] = move

        if self.end_index < self.nmatches*2:
            # Add pick to end of match_moves list now that this pick is available for selection in a later match
            self.available_moves[self.end_index] = move

        # Increment match and end index
        self.match_index += 1
        self.end_index += 1

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        if self.match_index < self.nmatches:
            # Return adjacent elements of the move list (next team pair in bracket)
            moves = self.available_moves[(2*self.match_index):(2*self.match_index+2)]

            # Always pick Kansas in first round
            if self.match_index == 0:
                moves = [self.teamlist[0]]
            # Always pick Oregon in first round
            if self.match_index == 8:
                moves = [self.teamlist[16]]
            # Always pick North Carolina in first round
            if self.match_index == 16:
                moves = [self.teamlist[32]]

            return moves

        else:
            # At the end of the game, the final action is to perform the random selection of chance nodes
            # that correspond to the true outcome of the matches as well as a randomly generated opponent,
            # both selected from the prior probilities
            if self.simulated_game is None:
                self.simulated_game = SingleEliminationBracket(self.teamlist).simulate_random_bracket()
                self.score = ncaa_custom_score(self.simulated_game, self.picks)
                self.opponent_picks = get_random_opponent_picks(self.teamlist)
                self.opponent_score = ncaa_custom_score(self.simulated_game, self.opponent_picks)
            return []

    def GetRandomMove(self):
        """ Get a random move according to the team ranking """
        teams = self.GetMoves()

        if teams == []:
            return []

        if len(teams) == 1:
            return teams[0]

        # Proabability that team 0 wins
        if random.random() < teams[0].win_probability(teams[1]):
            return teams[0]
        else:
            return teams[1]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        # Return 1 if this player won
        if self.score > self.opponent_score:
            return 1.0
        elif self.score == self.opponent_score:
            # For ties give 50% probability of win
            return random.choice((1.0, 0.0))
        else:
            return 0.0

    def __repr__(self):
        """ Don't need this - but good style.
        """
        return "Picks: {}".format(self.picks)


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + np.sqrt(2*np.log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(state.GetRandomMove())

        # Backpropagate
        while node is not None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print rootnode.TreeToString(0)
    else: print rootnode.ChildrenToString()

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited


def play_n_games(teamlist, picks, ngames, verbose=False):
    """ Plays n games with the given picks""" 
    wins = 0
    for i in range(ngames):
        sample_game = SingleEliminationBracket(teamlist).simulate_random_bracket()
        sample_opponent = get_random_opponent_picks(teamlist)
        if verbose:
            print '==My Score=='
        my_score = ncaa_custom_score(sample_game, picks, verbose)
        if verbose:
            print '==Other Score=='
        opponent_score = ncaa_custom_score(sample_game, sample_opponent, verbose)
        if my_score > opponent_score:
            wins += 1

    print "W/N: %d/%d = %.3f" % (wins, ngames, float(wins)/float(ngames))


def get_teams_from_db(db, teamlistfile):
    teamlist = []
    with open(teamlistfile, 'r') as tl:
        for line in tl:
            name, seed = line.split(',')
            seed = float(seed.strip())
            team = db[name]
            team.seed = seed
            teamlist.append(team)
    return teamlist


def UCTPlayGame(player_iter):
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    db = nct.load_db('ncaa2016.db')
    load_nate_silver(db, 'nate_silver_2016.txt')
    teamlist = get_teams_from_db(db, 'teamlist2016.txt')
    state = GameState(teamlist)
    while (state.GetMoves() != []):
        print str(state)
        m = UCT(rootstate=state, itermax=player_iter, verbose=False) # play with values for itermax and verbose = True
        print "Best Move: " + str(m) + "\n"
        state.DoMove(m)

    # Now play games with the selected player
    play_n_games(teamlist, state.picks, 10000)
    outfile = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S') + '_{}_iter'.format(player_iter) + '.json'
    with open(outfile, 'w') as f:
        json.dump([p.team_name for p in state.picks], f, indent=4)
    return state

if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """
    state = UCTPlayGame()
 