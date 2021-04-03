import getpass
import re
from datetime import datetime

import boto3
import pandas as pd
import sqlalchemy as sal
from tqdm import tqdm

from .utils import insert_with_progress

DISTRICTS = [3, 4, 5, 6, 7, 8, 10, 11, 12]
YEARS = [2018, 2019, 2020, 2021]

# S3
BUCKET_NAME = 'dse-grp3-capstone-data'

# RDS
ENDPOINT = 'capstone.clihskgj8i7s.us-west-2.rds.amazonaws.com'
USER = 'group3'
DB = 'db1'
PASSWORD = getpass.getpass('Enter database password')


def read_traffic(filename):
    df = pd.read_csv(f's3://{BUCKET_NAME}/{filename}', usecols=[0, 1, 7, 8, 9, 10, 11])
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


s3 = boto3.resource('s3')
bucket = s3.Bucket(BUCKET_NAME)
engine = sal.create_engine(f'postgresql://{USER}:{PASSWORD}@{ENDPOINT}/{DB}')
conn = engine.raw_connection()
rows = 0

for year in tqdm(YEARS, desc='Year'):
    for district in tqdm(DISTRICTS, desc='District', leave=False):
        prefix = f'caltrans/station_5min/{year}/d{district}'
        objects = bucket.objects.filter(Prefix=prefix)
        files = [o.key for o in objects if o.key[-7:] == '.txt.gz']
        files.sort()

        for f in tqdm(files, desc=f'District {district} ({year})', leave=False):
            # read data
            df = read_traffic(f)

            # insert data
            date = re.findall(r'\d{4}_\d{2}_\d{2}', f)[0]
            insert_with_progress(df, conn, 'traffic', pbar_desc=date)

            # track total rows inserted
            rows += len(df)

print(f'Total rows inserted: {rows}')
