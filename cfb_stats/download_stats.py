import requests
import pickle

start_date = '2009-08-01'
end_date = '2017-04-01'

url = 'http://sports.snoozle.net/api?league=cfb&fileType=json&statType=matchup&startDate={}&endDate={}'.format(start_date, end_date)


if __name__ == '__main__':
    r = requests.get(url)
    rslt = r.json()
    database_name = '{}_{}_raw.dat'.format(start_date[:4], end_date[:4])
    with open(database_name, 'w') as _db:
        pickle.dump(rslt['matchUpStats'], _db, pickle.HIGHEST_PROTOCOL)
