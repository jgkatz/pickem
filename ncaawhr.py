import sqlite3
import csv
import time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, String, Float, Integer, ForeignKey
from sqlalchemy.types import DateTime
from sqlalchemy import create_engine
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.orm import relationship, sessionmaker, backref
import wholehistory as whr
from datetime import datetime
import networkx as nx


Base = declarative_base()
Session = sessionmaker()
engine = None


class Team(Base):
    __tablename__ = 'team'
    id = Column(Integer, primary_key=True)
    team_name = Column(String)
    ranks = relationship("Rank",
                         backref=backref("team",
                                         order_by='Rank.date'))
    __tableargs__ = (UniqueConstraint('team_name'))


class Rank(Base):
    __tablename__ = 'rank'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    team_id = Column(Integer, ForeignKey('team.id'))
    rankmean = Column(Float)


class Match(Base):
    __tablename__ = 'match'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    team1_id = Column(Integer, ForeignKey('team.id'))
    team2_id = Column(Integer, ForeignKey('team.id'))
    team1 = relationship("Team", foreign_keys=team1_id,
                         backref=backref("t1matches",
                                         order_by='Match.date'))
    team2 = relationship("Team", foreign_keys=team2_id,
                         backref=backref("t2matches",
                                         order_by='Match.date'))
    rank1_id = Column(Integer, ForeignKey('rank.id'))
    rank2_id = Column(Integer, ForeignKey('rank.id'))
    rank1 = relationship("Rank", foreign_keys=rank1_id,
                         backref=backref("t1matches",
                                         order_by='Match.date'))
    rank2 = relationship("Rank", foreign_keys=rank2_id,
                         backref=backref("t2matches",
                                         order_by='Match.date'))

    score1 = Column(Float)
    score2 = Column(Float)
    __tableargs__ = (UniqueConstraint('date', 'team1', 'team2'))


def connect(dbfile):
    engine = create_engine('sqlite:///' + dbfile, echo=False)
    Session.configure(bind=engine)
    session = Session()
    return session


def create_db(session):
    Base.metadata.create_all(session.get_bind())


def load_from_csv(fname, session):
    G = nx.Graph()
    with open(fname, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        # Skip every other row because csv has both sides of match
        skip = False
        for row in reader:
            if skip:
                # Make sure pattern is followed
                assert row['Opponent'] == t1.team_name
                skip = False
                continue
            else:
                skip = True
            # try:
            # Create or find visitor and home team
            t1 = get_or_create(Team,
                               Team.team_name == row['Team'],
                               session)
            t1.team_name = row['Team']
            if not G.has_node(t1.team_name):
                G.add_node(t1.team_name)
            t2 = get_or_create(Team,
                               Team.team_name == row['Opponent'],
                               session)
            t2.team_name = row['Opponent']
            if not G.has_node(t2.team_name):
                G.add_node(t2.team_name)
            match = Match()
            match.date = datetime.strptime(row['Date'], '%m/%d/%Y')
            match.score1 = float(row['Team Score'])
            match.score2 = float(row['Opponent Score'])
            match.team1 = t1
            match.team2 = t2
            session.add(match)

            # Add a rank entity for each side of the match
            r1 = Rank()
            r1.rankmean = 0
            r1.date = match.date
            t1.ranks.append(r1)
            r2 = Rank()
            r2.rankmean = 0
            r2.date = match.date
            t2.ranks.append(r2)
            match.rank1 = r1
            match.rank2 = r2
            session.add(match)
            session.flush()

            # Add edge to graph
            G.add_edge(t1.team_name, t2.team_name)

            # except:
            #     print 'Skipping invalid row'
        session.commit()
        session.close()
    nx.write_gpickle(G, fname + '.graph')


def get_or_create(model, filters, session):
    object = session.query(model).filter(filters).first()
    if object is None:
        object = model()
    return object


def test_algorithm(sess):
    whrtool = whr.WholeHistoryRankUtil()
    t = sess.query(Team).filter(Team.team_name == 'Alabama').first()
    first = True
    for rank in t.ranks:
        e = whr.Epoch(first_epoch=first)
        e.set_natural_rank(rank.rankmean)
        e.date = rank.date
        # Clear first so only the initial epoch has first set to True
        first = False
        for t1match in rank.t1matches:
            result = whr.MatchResult(t1match.score1, t1match.score2, t1match.rank2.rankmean)
            e.add_match(result)
        for t2match in rank.t2matches:
            result = whr.MatchResult(t2match.score2, t2match.score1, t2match.rank1.rankmean)
            e.add_match(result)
        whrtool.add_epoch(e)
    import pdb; pdb.set_trace()
    whrtool.run_newton_step()


def update_single_team(team, n_newton=6):
    whrtool = whr.WholeHistoryRankUtil()
    first = True
    initrank = team.ranks[-1].rankmean
    for rank in team.ranks:
        e = whr.Epoch(first_epoch=first)
        e.set_natural_rank(rank.rankmean)
        e.date = rank.date
        # Clear first so only the initial epoch has first set to True
        first = False
        for t1match in rank.t1matches:
            result = whr.MatchResult(t1match.score1, t1match.score2, t1match.rank2.rankmean)
            e.add_match(result)
        for t2match in rank.t2matches:
            result = whr.MatchResult(t2match.score2, t2match.score1, t2match.rank1.rankmean)
            e.add_match(result)
        whrtool.add_epoch(e)
    # Compute initial likelihood
    s2 = whrtool.compute_sigma_2()
    logp_initial = whrtool.log_likelihood(s2)
    for i in range(n_newton):
        whrtool.run_newton_step()
    for ix, rankval in enumerate(whrtool.get_rank_vec()):
        team.ranks[ix].rankmean = rankval
    logp_final = whrtool.log_likelihood(s2)
    endrank = team.ranks[-1].rankmean
    return endrank - initrank, logp_final - logp_initial, logp_final


def run_pass(sess, n_newton=6):
    teams = sess.query(Team)
    ll = 0
    maxdelta = 0
    maxteam = None
    for t in teams:
        rankdelta, lldelta, ll_i = update_single_team(t, n_newton)
        if abs(rankdelta) > abs(maxdelta):
            maxdelta = rankdelta
            maxteam = t
        ll += ll_i
        print 'updated %s, delta %s' % (t.team_name, rankdelta)
    print 'Total likelihood: %f, max delta %f' % (ll, maxdelta)
    print 'Max delta team: %s' % maxteam.team_name
    return ll, maxdelta, maxteam.team_name


def get_team_rank(team, sess):
    return sess.query(Team).filter(Team.team_name==team).first().ranks[-1].rankmean
