import os
import sys
from concurrent import futures
from functools import partial

import pandas as pd

import stwits_data_loader

pd.set_option('display.max_columns', 100)

working_directory = os.getcwd()
os.chdir(working_directory)
sys.path.append(working_directory)


def msgs_load_worker(symbol_info, db_name, token):
    print("\r\nRetrieving all messages for '{}'".format(symbol_info["title"]))
    data_provider = stwits_data_loader.MongoDBDataLoader(db_name=db_name, token=token, min_msg_id=180000000,
                                                         host="127.0.0.1", port=27017)
    data_provider.retrieve_all_messages_from_stocktwits(symbol_info["symbol_name"])
    return 0


def main():
    # with open("stwits_data_loader/resources/access_token.txt", "r") as tokenFile:
    #     token = tokenFile.readline().rstrip()
    token = ''
    db_name = "stocktwits_msgs"
    data_provider = stwits_data_loader.MongoDBDataLoader(db_name=db_name, token=token, min_msg_id=180000000,
                                                         host="127.0.0.1", port=27017)
    symbol_infos = data_provider.retrieve_symbol_infos_filtered_by_mkt_cap_async()

    with futures.ProcessPoolExecutor(2) as executor:
        results = list(executor.map(partial(msgs_load_worker, db_name=db_name, token=token), symbol_infos))
    results_all = list(results)
    print(sum(results_all))


if __name__ == '__main__':
    main()


