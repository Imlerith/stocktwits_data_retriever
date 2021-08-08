import csv
import os
import sys
from datetime import datetime as dt, timedelta
import itertools
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import pymongo
import psycopg2 as pg

import stwits_data_loader

pd.set_option('display.max_columns', 100)

working_directory = os.getcwd()
os.chdir(working_directory)
sys.path.append(working_directory)

db_name = "stocktwits_msgs"
host = "127.0.0.1"
port = 27017
password = ""
data_path = "/new/drive/stocktwits_data/"


def slice_delta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


def get_pairs(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))


def execute_pg_query(conn, query_type="insert", pg_query="", insert_file="abc.csv", get_df=False):
    assert query_type in ["select", "insert"], "query_type parameter can be either 'select' or 'insert'"
    assert pg_query.startswith("SELECT") if query_type == "select" else True, \
        "For query_type parameter 'select' query must start with 'SELECT'"
    """ Connect to the PostgreSQL database server """
    try:
        # --- create a cursor
        cur = conn.cursor()

        # --- fetch the data
        if query_type == "select":
            # --- execute the query
            cur.execute(pg_query)
            # --- return the result
            if not get_df:
                result = cur.fetchall()
            else:
                result = pd.read_sql_query(pg_query, con=conn)
            # --- close the communication with the server
            cur.close()
            # --- return the result
            return result
        else:
            with open(insert_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    cur.execute(
                        "INSERT INTO all_messages VALUES (%s, %s, %s, %s, %s, %s, %s)", row
                    )
            conn.commit()
    except (Exception, pg.DatabaseError) as error:
        # --- if error, return None
        print(error)
        return None


# ----- Read messages from database into a file
# --- (a) read crypto symbols
with open("stwits_data_loader/resources/access_token.txt", "r") as tokenFile:
    token = tokenFile.readline().rstrip()
data_provider = stwits_data_loader.MongoDBDataLoader(db_name=db_name, token=token, min_msg_id=180000000,
                                                     host=host, port=port)
symbol_infos = data_provider.retrieve_symbol_infos_filtered_by_mkt_cap_async()
crypto_symbols_set = {symbol_info["symbol_name"] for symbol_info in symbol_infos}

# --- (b) read min and max datetimes
pipeline = [{'$group': {'_id': {}, 'min': {'$min': '$created_at'}, 'max': {'$max': '$created_at'}}}]
client = pymongo.MongoClient(host, port)
database = client[db_name]
collection = database["Messages"]
min_max_datetimes = list(collection.aggregate(pipeline))
min_dt, max_dt = min_max_datetimes[0]['min'], min_max_datetimes[0]['max']

# here check if the Postgres db not empty
conn = pg.connect(host="localhost", database="stocktwits", user="postgres", password="postgres")
query_min_time = "SELECT MAX(timestamp) " \
                 "FROM all_messages"
min_dt_db = execute_pg_query(conn, query_type="select", pg_query=query_min_time)
if min_dt_db is not None:
    min_dt_db = min_dt_db[0][0]
    min_dt = min_dt_db  # last timestamp in Postgres db

# --- (c) get pairs of dates between which to load the data
all_dates = list()
for dtime in slice_delta(min_dt, max_dt, timedelta(days=30)):
    all_dates.append(dtime)
all_dates.append(max_dt)
dates_pairs = get_pairs(all_dates)

# --- (d) read and save data in chunks
separate_files_path = glob(f"{data_path}separate_files/*.csv")
if separate_files_path:
    for separate_file_path in separate_files_path:
        os.remove(separate_file_path)
for date_pair in tqdm(dates_pairs):
    all_messages = data_provider.read_messages_from_database(start_dt=date_pair[0], end_dt=date_pair[1])
    data_cols = ['id', 'user_id', 'symbol', 'body', 'sentiment', 'timestamp']
    messages_df = pd.DataFrame(index=range(len(all_messages)), columns=data_cols)
    for i, msg in enumerate(all_messages):
        msg_symbol = msg['symbols'][0]['symbol']
        if msg_symbol not in crypto_symbols_set:
            continue
        else:
            if msg['entities']['sentiment'] is None:
                msg_sentiment = 999
            else:
                msg_sentiment_ = msg['entities']['sentiment']['basic']
                msg_sentiment = 1 if msg_sentiment_ == 'Bullish' else 0
            msg_id = msg['id']
            msg_user_id = msg['user']['id']
            msg_body = msg['body']
            msg_timestamp = msg['created_at']
            messages_df.loc[i, data_cols] = [msg_id, msg_user_id, msg_symbol, msg_body, msg_sentiment, msg_timestamp]
    messages_df = messages_df.dropna(subset=['id']).reset_index(drop=True)
    date_start_str = '_'.join(date_pair[0].__str__().split(' '))
    date_end_str = '_'.join(date_pair[1].__str__().split(' '))
    date_start_end_str = f"{date_start_str}--{date_end_str}"
    messages_df.to_csv(f"{data_path}separate_files/msgs_{date_start_end_str}.csv")

# --- (d) read new data from chunks and save them in one file
new_messages_df = pd.DataFrame(columns=['id', 'user_id', 'symbol', 'body', 'sentiment', 'timestamp'])
for filename in tqdm(os.listdir(f"{data_path}separate_files")):
    print(filename)
    full_path = os.path.join(f"{data_path}separate_files", filename)
    messages_df = pd.read_csv(full_path, index_col=0)
    new_messages_df = new_messages_df.append(messages_df, ignore_index=True)

new_messages_df = new_messages_df.drop_duplicates(subset=['id'])
new_messages_df = new_messages_df.sort_values(by=['timestamp']).reset_index(drop=True)
new_messages_df = new_messages_df.assign(sentiment_type='')
new_messages_df['sentiment_type'] = new_messages_df['sentiment']\
    .apply(lambda x: 'real' if not np.isnan(x) else 'predicted')
new_messages_df = new_messages_df[new_messages_df['timestamp'] !=
                                  min_dt.strftime("%Y-%m-%d %H:%M:%S")].reset_index(drop=True)
new_messages_df.to_csv(f"{data_path}new_messages_df.csv", index=False)

# --- (e) put the new data into Postgres database
execute_pg_query(conn, insert_file=f"{data_path}new_messages_df.csv")

# --- (f) export all available data from Postgres for index re-construction
query_all_data = f"SELECT * FROM all_messages"
all_data_from_pg = execute_pg_query(conn, query_type="select", pg_query=query_all_data, get_df=True)
all_data_from_pg.to_csv(f"{data_path}all_messages_df_updated.csv", index=False)

# # --- (g) add messages from Postgres from the first day in the new data to the new data
# #         (to re-calibrate the sentiment for this day)
# first_day_start = new_messages_df['timestamp'][0][:10] + " 00:00:00"
# first_day_end = new_messages_df['timestamp'][0][:10] + " 23:59:59"
# query_first_day = f"SELECT * FROM all_messages " \
#                   f"WHERE timestamp >= '{first_day_start}' " \
#                   f"AND timestamp <= '{first_day_end}'"
# first_day_from_pg = execute_pg_query(conn, query_type="select", pg_query=query_first_day, get_df=True)
# new_messages_df = pd.concat([first_day_from_pg, new_messages_df], axis=0, ignore_index=True)\
#                     .drop_duplicates(subset=['id'])
# new_messages_df.to_csv(f"{data_path}new_messages_df.csv", index=False)

