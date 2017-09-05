import sqlite3
import csv
import time
from dateutil import parser
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, String, Float, Integer, ForeignKey
from sqlalchemy.types import DateTime, Boolean
from sqlalchemy import create_engine
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.orm import relationship, sessionmaker, backref
import json
from contextlib import contextmanager


Base = declarative_base()
Session = sessionmaker()
engine = None


class Team(Base):
    __tablename__ = 'team'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    game_stats = relationship('TeamGameStats',
                              backref='team',
                              lazy='dynamic')
    __tableargs__ = (UniqueConstraint('name'))

    def __repr__(self):
        return '<Team {}>'.format(self.name)


class TeamGameStats(Base):
    """Statistics for a team for a single game."""
    __tablename__ = 'team_game_stats'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    team_id = Column(Integer, ForeignKey('team.id'))
    stats = Column(String)
    home = Column(Boolean)

    def get_jstats(self):
        return json.loads(self.stats)

    def set_jstats(self, val):
        self.stats = json.dumps(val)


class Game(Base):
    __tablename__ = 'game'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    home_stats_id = Column(Integer, ForeignKey('team_game_stats.id'))
    vis_stats_id = Column(Integer, ForeignKey('team_game_stats.id'))
    home_stats = relationship("TeamGameStats",
                              foreign_keys=home_stats_id)
    vis_stats = relationship("TeamGameStats",
                             foreign_keys=vis_stats_id)
    __tableargs__ = (UniqueConstraint('date', 'home_stats_id', 'vis_stats_id'))


def connect(dbfile):
    engine = create_engine('sqlite:///' + dbfile, echo=False)
    Session.configure(bind=engine)
    session = Session()
    return session


def create_db(session):
    Base.metadata.create_all(session.get_bind())


def get_or_create(model, filters, session):
    object = session.query(model).filter(filters).first()
    if object is None:
        object = model()
    return object


@contextmanager
def session_manager(dbfile):
    """Provide a transactional scope around a series of operations."""
    session = connect(dbfile)
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def import_raw(json_data, dbfile):
    invalid = 0
    with session_manager(dbfile) as session:
        create_db(session)
        for match_item in json_data:
            home_name = match_item['homeTeamName']
            vis_name = match_item.get('visTeamName')
            if not vis_name:
                invalid += 1
                continue
            assert home_name != vis_name

            home_team = get_or_create(Team, Team.name == home_name, session)
            home_team.name = home_name
            vis_team = get_or_create(Team, Team.name == vis_name, session)
            vis_team.name = vis_name

            game_date = parser.parse(match_item['date'])

            home_stats = TeamGameStats()
            home_stats.team = home_team
            home_stats.stats = json.dumps(match_item['homeStats'])
            home_stats.date = game_date

            vis_stats = TeamGameStats()
            vis_stats.team = vis_team
            vis_stats.stats = json.dumps(match_item['visStats'])
            vis_stats.date = game_date

            game = Game()
            game.date = game_date
            game.home_stats = home_stats
            game.vis_stats = vis_stats

            session.add(game)

    print '{} invalid matches'.format(invalid)
