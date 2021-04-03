import getpass

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


def read_station_meta(filename):
    df = pd.read_csv(f's3://{BUCKET_NAME}/{filename}', sep='\t', usecols=range(14))
    df.columns = [
        'sid',
        'fwy',
        'direc',
        'district',
        'county',
        'city',
        'state_pm',
        'abs_pm',
        'latitude',
        'longitude',
        'length',
        'stype',
        'lanes',
        'name',
    ]
    df['city'] = df['city'].astype('Int64')

    return df


s3 = boto3.resource('s3')
bucket = s3.Bucket(BUCKET_NAME)
engine = sal.create_engine(f'postgresql://{USER}:{PASSWORD}@{ENDPOINT}/{DB}')
conn = engine.raw_connection()
rows = 0

for district in tqdm(DISTRICTS, desc='District'):
    files = []

    # get all files for the current district
    for year in YEARS:
        prefix = f'caltrans/meta/{year}/d{district}'
        files.extend([o.key for o in bucket.objects.filter(Prefix=prefix)])

    # sort files in reverse chronological order
    files.sort(reverse=True)

    # read files into memory
    desc = f'District {district}'
    pbar = tqdm(files, desc=f'Read data: {desc}', leave=False)
    df_list = [read_station_meta(f) for f in pbar]

    # combine stations into a single dataframe and drop duplicates (keeping most recent)
    df = pd.concat(df_list)
    df = df.drop_duplicates(subset='sid', keep='first')

    # insert into database
    insert_with_progress(df, conn, 'pemslocs', pbar_desc=desc, sep=';')

    # track total rows inserted
    rows += len(df)

print(f'Total rows inserted: {rows}')
