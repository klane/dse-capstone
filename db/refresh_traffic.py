import getpass
import glob
import io
import re
from datetime import datetime

import pandas as pd
import sqlalchemy as sal
from tqdm import tqdm

ENDPOINT = 'capstone.clihskgj8i7s.us-west-2.rds.amazonaws.com'
USER = 'group3'
DB = 'db1'
PASSWORD = getpass.getpass('Enter database password')


def chunker(seq, size):
    # from http://stackoverflow.com/a/434328
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_traffic(filename):
    df = pd.read_csv(filename, header=None)
    df = df.iloc[:, [0, 1, 7, 8, 9, 10, 11]]
    df.columns = [
        'timestamp',
        'station',
        'samples',
        'pct_observed',
        'total_flow',
        'avg_occupancy',
        'avg_speed',
    ]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['total_flow'] = df['total_flow'].astype('Int64')

    date = re.findall(r'\d{4}_\d{2}_\d{2}', filename)[0]
    date = datetime.strptime(date, '%Y_%m_%d').date()
    df = df[df['timestamp'].dt.date == date]

    return df


def insert_with_progress(df, conn, sep=','):
    chunksize = int(len(df) / 10)
    rows = 0

    with tqdm(total=len(df), desc='Current file', leave=False) as pbar:
        for cdf in chunker(df, chunksize):
            cursor = conn.cursor()
            fbuf = io.StringIO()
            cdf.to_csv(fbuf, index=False, header=False, sep=sep)
            fbuf.seek(0)
            cursor.copy_from(fbuf, 'traffic', sep=sep, null='')
            conn.commit()
            cursor.close()
            pbar.update(chunksize)
            rows += len(cdf)

    return rows


traffic_files = glob.glob('caltrans/station_5min/2021/d*/*.txt.gz')
engine = sal.create_engine(f'postgresql://{USER}:{PASSWORD}@{ENDPOINT}/{DB}')
conn = engine.raw_connection()
rows = 0
traffic_files.reverse()

for f in tqdm(traffic_files, desc='All files'):
    try:
        df = read_traffic(f)
    except Exception as e:
        print(f'Error reading {f}')
        print(e)
        continue

    try:
        rows += insert_with_progress(df, conn)
    except Exception:
        print(f'Error inserting {f}')
        continue

print(f'Total rows: {rows}')
