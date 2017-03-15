import numpy as np
import csv
import os
from trueskill import Rating, rate_1vs1, quality_1vs1, choose_backend
import pickle
from ncaa_bracket import AbstractTeam
cdf, pdf, ppf = choose_backend(None)


class Match(object):
    def __init__(self, winner, loser, draw):
        self.winner = winner
        self.loser = loser
        self.draw = draw


class Team(AbstractTeam):
    def __init__(self, team_name, rating, seed=None):
        self.team_name = team_name
        self.rating = rating

    def serialize(self):
        return {
            'team_name': self.team_name,
            'rating_mu': self.rating.mu,
            'rating_sigma': self.rating.sigma
        }

    # From https://github.com/sublee/trueskill/issues/1
    def win_probability(self, other):
        """Compute win probability from CDF of distributions corresponding to prob that self is better than b."""
        rA = self.rating
        rB = other.rating
        deltaMu = rA.mu - rB.mu
        rsss = np.sqrt(rA.sigma**2 + rB.sigma**2)
        return cdf(deltaMu/rsss)

    def __repr__(self):
        return self.team_name

# Threshold at which score difference is considered a draw
draw_threshold = 0


def load_massey(basedir):
    db = {
        'id_to_team': {},
        'name_to_team': {},
        'matches': []
    }
    
    teamlistfile = os.path.join(basedir, 'teams.csv')
    with open(teamlistfile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for team_id, team_name in reader:
            team_id = team_id.strip()
            team_name = team_name.strip()

            # Create a trueskill rating for each team and two ways of accessing
            team_rating = Rating()
            db['id_to_team'][team_id] = Team(team_name, team_rating)
            db['name_to_team'][team_name] = db['id_to_team'][team_id]

    matchlistfile = os.path.join(basedir, 'games.csv')
    with open(matchlistfile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # Unpack
            datestr, team_a_id, team_a_home, team_a_score = row[1:5]
            team_b_id, team_b_home, team_b_score = row[5:8]

            # Cast
            team_a_id = team_a_id.strip()
            team_b_id = team_b_id.strip()
            team_a_score = int(team_a_score)
            team_b_score = int(team_b_score)

            # Look up teams in db
            if team_a_score > team_b_score:
                winner = team_a_id
                loser = team_b_id
            elif team_b_score > team_a_score:
                winner = team_b_id
                loser = team_a_id
            else:
                raise Exception('Draw! Impossible!')

            draw = False
            if abs(team_a_score - team_b_score) < draw_threshold:
                draw = True

            db['matches'].append(Match(winner, loser, draw))

    return db


def save_db(db, file):
    with open(file, 'wb') as f:
        out_d = {}
        for k in db['name_to_team']:
            out_d[k] = db['name_to_team'][k].serialize()
        pickle.dump(out_d, f, protocol=2)


def load_db(file):
    with open(file, 'rb') as f:
        result = pickle.load(f)
        out_db = {}
        for k, v in result.iteritems():
            out_db[k] = Team(v['team_name'], Rating(mu=v['rating_mu'], sigma=v['rating_sigma']))
    return out_db


def process_matches(db):
    """Run through all matches sequentially and update ranks."""
    for ix, match in enumerate(db['matches']):
        winner = db['id_to_team'][match.winner]
        loser = db['id_to_team'][match.loser]

        # Run rating update
        winner.rating, loser.rating = rate_1vs1(winner.rating, loser.rating, match.draw)

        if ix % 10 == 0:
            print 'Finished match %d' % (ix + 1)


def get_sorted_teams(db):
    # Sort by mean - 2-sigma
    team_by_rating = sorted(db['id_to_team'].items(), key=lambda x: x[1].rating.mu - x[1].rating.sigma*2.0)
    return team_by_rating
