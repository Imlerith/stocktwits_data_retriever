import os
import sys
import asyncio

import pandas as pd
import aiohttp

import stwits_data_loader

pd.set_option('display.max_columns', 100)

working_directory = os.getcwd()
os.chdir(working_directory)
sys.path.append(working_directory)


async def update_messages_from_stocktwits_async():
    async with aiohttp.ClientSession() as session:
        with open("stwits_data_loader/resources/access_token.txt", "r") as tokenFile:
            token = tokenFile.readline().rstrip()
        db_name = "stocktwits_msgs"
        data_provider = stwits_data_loader.MongoDBDataLoader(db_name=db_name, token=token, min_msg_id=180000000,
                                                             host="127.0.0.1", port=27017)
        symbol_infos = data_provider.retrieve_symbol_infos_from_stocktwits()

        tasks = []
        for symbol_info in symbol_infos:
            max_symbol_id = data_provider.get_max_msg_id_for_symbol(symbol_info["symbol_name"])
            tasks.append(asyncio.ensure_future(data_provider.retrieve_all_messages_from_stocktwits_async(
                session, max_symbol_id, symbol_info["symbol_name"])))

        original_tasks = await asyncio.gather(*tasks)
        for task_ in original_tasks:
            print(task_)


asyncio.run(update_messages_from_stocktwits_async())


