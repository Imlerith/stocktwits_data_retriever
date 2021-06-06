"""Keep database, lexicons and plots up-to-date.

    Functions:
    update_all -- Update everything.
    update_database -- Update list of currencies and messages.
    make_lexicon -- Create and save unfiltered lexicon.
    plots_and_tables -- Create and save all figures and tables.
    sample_stats -- Print database stats that should be updated in article.
"""
import os
import sys

import pandas as pd
from tqdm import tqdm

import stwits_data_loader

pd.set_option('display.max_columns', 100)

working_directory = os.getcwd()
os.chdir(working_directory)
sys.path.append(working_directory)


def update_messages_from_stocktwits(data_provider):
    """Update Messages From StockTwist to Sqlite database

    Update list of constituents, download all related messages that haven't
    been downloaded yet, and create the timeline.
    When StockTwits API rate limit is hit, wait until requests are allowed
    again and keeps going until everything is up-to-date.
    """

    # --- if reading the latest symbol infos from the Web service fails, read from a file instead
    try:
        symbol_infos = data_provider.retrieve_symbol_with_dot_x_infos_from_stocktwits()
    except Exception as e:
        print("Retrive lastest symbolInfos from StockTwists' Web service fail. Error: {}", e)
        print("Read from file instead.", e)
        symbol_infos = data_provider.read_symbol_infos_from_file()

    for symbol_info in symbol_infos:
        print("\r\nRetriving all messages for '{}'".format(symbol_info["title"]))
        # data_provider.retrieve_all_messages_from_stocktwits(symbol_info["symbol_name"], wait_counter=1)
        data_provider.retrieve_all_messages_from_stocktwits_async(symbol_info["symbol_name"])


def main():
    with open("stwits_data_loader/resources/access_token.txt", "r") as tokenFile:
        token = tokenFile.readline().rstrip()
    db_name = "stocktwits_msgs"
    data_provider = stwits_data_loader.MongoDBDataLoader(db_name=db_name, token=token, min_msg_id=180000000,
                                                         host="127.0.0.1", port=27017)
    update_messages_from_stocktwits(data_provider)

    all_messages = data_provider.read_messages_from_database()
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

    messages_df = messages_df.dropna().reset_index(drop=True)
    messages_df.to_csv('messages_df.csv')


if __name__ == '__main__':
    main()


