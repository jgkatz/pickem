from ncaa_cfb_db import session_manager, Team, Game, TeamGameStats
from sqlalchemy import asc, desc
import datetime
from trueskill import Rating, rate_1vs1, quality_1vs1, choose_backend
from collections import OrderedDict
import csv
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from functools import partial

stat_db = 'cfb_stats/ncaa_2009_2017_stats.db'


def get_prev_stats(team, date, min_date=None, limit=None):
    """Get game for team prior to date."""
    q = team.game_stats.filter(TeamGameStats.date < date)
    q = q.order_by(desc(TeamGameStats.date))
    if min_date:
        q = q.filter(TeamGameStats.date >= min_date)
    if limit is None:
        q = q.limit(1)
        return q.one_or_none()

    q = q.limit(3)
    return q.all()


def get_seasons(start_year, end_year):
    season_ranges = zip(range(start_year, end_year),
                        range(start_year + 1, end_year + 1))

    return [
        (datetime.datetime(season_start, 7, 1, 0, 0),
         datetime.datetime(season_end, 4, 1, 0, 0))
        for season_start, season_end in season_ranges
    ]


def add_trueskill():
    """Append TrueSkill ratings to all stats in database.

    Ratings are computed on a season by season basis.
    """

    new_season = True
    start_year = 2009
    end_year = 2017
    with session_manager(stat_db) as session:
        for min_date, max_date in get_seasons(start_year, end_year):
            all_games = (
                session.query(Game)
                .filter(Game.date >= min_date)
                .filter(Game.date <= max_date)
                .order_by(asc(Game.date))
                .all()
            )
            for game in all_games:

                # Try to get the teams' ratings from the previous game
                ratings = {}
                for home_or_vis in ('home', 'vis'):
                    team = getattr(game, home_or_vis + '_stats').team
                    prev_stats = get_prev_stats(team, game.date, min_date)
                    if prev_stats is None:
                        # If no rating exists, initialize one
                        rating = Rating()
                    else:
                        stats_parsed = prev_stats.get_jstats()
                        rating = Rating(mu=stats_parsed['trueskill_mu'],
                                        sigma=stats_parsed['trueskill_sigma'])
                    ratings[home_or_vis] = rating

                # Find out who won
                home_score = game.home_stats.get_jstats()['score']
                vis_score = game.vis_stats.get_jstats()['score']

                # Run trueskill update
                if home_score > vis_score:
                    home_rating, vis_rating = rate_1vs1(
                        ratings['home'], ratings['vis'])
                else:
                    vis_rating, home_rating = rate_1vs1(
                        ratings['vis'], ratings['home'])
                new_ratings = {
                    'home': home_rating,
                    'vis': vis_rating
                }

                # Save back to DB
                for home_or_vis in ('home', 'vis'):
                    team_stats = getattr(game, home_or_vis + '_stats')
                    jstats = team_stats.get_jstats()
                    jstats['trueskill_mu'] = new_ratings[home_or_vis].mu
                    jstats['trueskill_sigma'] = new_ratings[home_or_vis].sigma
                    team_stats.set_jstats(jstats)

def generate_dataset():
    """Iterate through all games and store input vectors.

    1. Load every game,
    2. Find both involved teams
    3. Query past 3 games.
    4. Store stats for all 3 games for both teams

    """

    stat_fields = (
        'fourthDownAtt',
        'rushYds',
        'passAtt',
        'thridDownAtt',
        'fourthDownConver',
        'passComp',
        'score',
        'rushAtt',
        'timePoss',
        'fumblesLost',
        'passYds',
        'penaltYds',
        'thirdDownConver',
        'penalties',
        'interceptionsThrown',
        'firstDowns',
        'trueskill_mu',
        'trueskill_sigma'
    )

    dataset_file = 'all_game_data.csv'

    start_year = 2009
    end_year = 2017
    num_prev = 3
    all_entries = []
    with session_manager(stat_db) as session:
        for min_date, max_date in get_seasons(start_year, end_year):
            all_games = (
                session.query(Game)
                .filter(Game.date >= min_date)
                .filter(Game.date <= max_date)
                .order_by(asc(Game.date))
                .all()
            )
            print 'Year {}'.format(min_date)
            for game in all_games:
                skip = False
                all_stats = OrderedDict()
                for home_or_vis in ('home', 'vis'):
                    team = getattr(game, home_or_vis + '_stats').team
                    prev_stats = get_prev_stats(team, game.date, min_date, limit=3)
                    # If there isn't at least one prev game, skip
                    nprev = len(prev_stats)
                    if not nprev:
                        skip = True
                        break

                    # Duplicate last element if there are not enough
                    for _ in range(0, 3 - nprev):
                        prev_stats.append(prev_stats[-1])

                    for ix, prev_stat in enumerate(prev_stats):
                        jstats = prev_stat.get_jstats()
                        for stat_field in stat_fields:
                            full_key = '{}_{}_{}'.format(home_or_vis, ix, stat_field)
                            all_stats[full_key] = jstats[stat_field]
                if skip:
                    continue

                # Find out who won
                home_score = game.home_stats.get_jstats()['score']
                vis_score = game.vis_stats.get_jstats()['score']

                # Run trueskill update
                if home_score > vis_score:
                    all_stats['home_won'] = 1.
                else:
                    all_stats['home_won'] = 0.

                all_entries.append(all_stats)
    with open(dataset_file, 'wb') as _f:
        writer = csv.DictWriter(_f, all_entries[0].keys())
        writer.writeheader()
        writer.writerows(all_entries)


def load_dataset():
    df = pandas.read_csv('all_game_data.csv')
    dataset = df.values
    in_cols = dataset.shape[1] - 1
    X = dataset[:, 0:in_cols].astype(float)
    # Normalize
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    Y = dataset[:, -1]

    return X, Y


def train():
    X, Y = load_dataset()
    seed = 7
    numpy.random.seed(seed)

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    estimator = KerasClassifier(build_fn=partial(create_model, X), nb_epoch=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

    print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def create_model(X):
    model = Sequential()
    cols = X.shape[1]
    model.add(Dense(cols, input_dim=cols, kernel_intializer='normal', activation='relu'))
    model.add(Dense(1, kernel_intializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model