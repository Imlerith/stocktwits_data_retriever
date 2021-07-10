import os
import sys
from datetime import datetime as dt

import pandas as pd
from tqdm import tqdm

import stwits_data_loader

pd.set_option('display.max_columns', 100)

working_directory = os.getcwd()
os.chdir(working_directory)
sys.path.append(working_directory)


# ----- Read messages from database into a file
with open("stwits_data_loader/resources/access_token.txt", "r") as tokenFile:
    token = tokenFile.readline().rstrip()
db_name = "stocktwits_msgs"
data_provider = stwits_data_loader.MongoDBDataLoader(db_name=db_name, token=token, min_msg_id=180000000,
                                                     host="127.0.0.1", port=27017)

all_messages = data_provider.read_messages_from_database(start_dt=dt(2021, 1, 1), end_dt=dt(2021, 1, 30))
data_cols = ['id', 'user_id', 'message', 'sentiment', 'timestamp']
messages_df = pd.DataFrame(index=range(len(all_messages)), columns=data_cols)
for i, msg in tqdm(enumerate(all_messages)):
    try:
        msg_id = msg['id']
        msg_user_id = msg['user']['id']
        msg_body = msg['body']
        msg_sentiment = msg['entities']['sentiment']['basic']
        msg_timestamp = msg['created_at']
        messages_df.loc[i, data_cols] = [msg_id, msg_user_id, msg_body, msg_sentiment, msg_timestamp]
        # print(f"{msg_sentiment}: {msg_body}")
    except TypeError:
        continue

del all_messages
# messages_df = messages_df.dropna().reset_index(drop=True)
messages_df.to_csv('messages_df.csv')

