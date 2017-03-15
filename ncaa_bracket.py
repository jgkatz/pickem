import math
import random


class AbstractTeam(object):

    def win_probability(self, other_team):
        """Should return the probability of winning against the supplied team"""
        pass


def ncaa_custom_score(match_list, picks, verbose=False):
    """Compute the score according to the GNC 2016 rules.

    Args:
        match_list: List of match outcomes.
        picks: List of picks, same length as matches.
        verbose: Print matches.
    """
    seed_multiplier_by_round = {
        0: 1.0,
        1: 1.0,
        2: 2.0,
        3: 2.0,
        4: 3.0,
        5: 3.0
    }
    tpl = '{:<2} {:<2} {:<4} {:<10.10} {:<10.10} {:<8} {:<8}'
    score = 0.
    if verbose:
        print tpl.format('MT', 'RD','MULT','Game','Pick','SeedB','Score')
    for ix, match in enumerate(match_list):
        # 0-based round index
        round = 5 - int(math.log(63-ix, 2))
        multiplier = 2.**round
        seed_bonus = 0
        if match.winner.team_name == picks[ix].team_name:
            score += multiplier
            # Apply seed bonus if the predicted winner is an underdog.
            # Note that winner.seed - team.seed will be 0 for
            # one of the teams and either positive if the winning team
            # is an underdog or negative if the winning team is expected.
            # A multiplier is also applied
            seed_bonus = max(( 
                float(match.winner.seed - match.team_a.seed),
                float(match.winner.seed - match.team_b.seed),
                0.
            ))*seed_multiplier_by_round[round]
            score += seed_bonus
        if verbose:
            print tpl.format(ix+1, round, multiplier, match.winner.team_name, picks[ix].team_name, seed_bonus, score)
    return score


class Match(object):
    def __init__(self):
        self.parent = None
        self.team_a = None
        self.team_b = None
        self.winner = None

    def __repr__(self):
        marker_a = ''
        marker_b = ''
        if self.winner == self.team_a:
            marker_a = '*'
        if self.winner == self.team_b:
            marker_b = '*'
        if self.winner:
            return '%s%s vs %s%s' % (str(self.team_a), marker_a, str(self.team_b), marker_b)


class SingleEliminationBracket(object):

    def __init__(self, team_vector):
        """Initializes the class with a vector of team objects assumed to be in bracket order with
        an even number of elements"""

        # Create list of teams using the database
        self._teams = team_vector

        # Number of teams
        self._nteams = len(team_vector)

        # Reset match list
        self.reset()

    def reset(self):
        # Initialize match vector to number of matches in single elimination (n-1)
        self._matches = [Match() for i in range(self._nteams - 1)]

        for ix, t in enumerate(self._teams):
            # Set up initial matches
            if ix % 2 == 0:
                self._matches[ix/2].team_a = t
            else:
                self._matches[ix/2].team_b = t

        # Finish creating remaining matches and set parent relationships
        match_ix = 0
        end_ix = self._nteams/2
        while end_ix < len(self._matches):
            # Set match index and neighbor to point to next available match in the list
            self._matches[match_ix].parent = end_ix
            self._matches[match_ix + 1].parent = end_ix

            # Increment the match and index pointer
            match_ix += 2
            end_ix += 1

    def simulate_random_match(self, match):
        """Draws randomly from the weighted binary distribution of two teams and returns
        the winning team"""

        if random.random() <= match.team_a.win_probability(match.team_b):
            return match.team_a
        else:
            return match.team_b

    def simulate_random_bracket(self):
        """Simulates entire bracket drawing each match outcome from the pairwise team
        probabilities."""

        # Reset all the match states
        self.reset()

        # To simulate, just step through the match list in order and set the winner of
        # the parent appropriately
        for match in self._matches:
            match.winner = self.simulate_random_match(match)
            if match.parent is not None:
                parent = self._matches[match.parent]
                # Figure out which to set by which match is empty
                if parent.team_a is None:
                    parent.team_a = match.winner
                else:
                    parent.team_b = match.winner

        return self._matches
