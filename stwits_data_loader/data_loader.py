from abc import ABC, abstractmethod
import os
import json
import time
import datetime

import requests
import io
import asyncio
from importlib import util

import pandas as pd
import aiohttp
from bs4 import BeautifulSoup as BS

from stwits_data_loader import StockTwitsClient


class DataLoader(ABC):
    """The base class which implements functionality to retrieve messages from StockTwits"""

    resource_name = "stwits_data_loader.resources"
    symbols_info_filename = "cryptos_infos.json"
    filtered_symbols_info_filename = "filtered_cryptos_infos.json"
    spec = util.find_spec(resource_name)
    path = spec.submodule_search_locations[0]

    def __init__(self, token: str, min_msg_id: int = 200000000):
        self._token = token
        self._min_msg_id = min_msg_id
        self._stocktwits_api_symbol_infos_url = "http://api.stocktwits.com/symbol-sync/symbols.csv"
        self._api = StockTwitsClient()
        self._api.login(self._token)
        self._symbols_info_filepath = os.path.join(self.path, self.symbols_info_filename)
        self._filtered_symbols_info_filepath = os.path.join(self.path, self.filtered_symbols_info_filename)
        self._symbol_infos = self.retrieve_symbol_infos_from_stocktwits()

    def retrieve_symbol_infos_from_stocktwits(self):
        """
        Retrieve information on coins from StockTwists (filtered by .X names applied)
        """
        file_time = 0
        # --- update the symbol list from stocktwits every 10 days
        if os.path.exists(self._symbols_info_filepath):
            file_time = os.path.getctime(self._symbols_info_filepath)

        # --- get the latest symbol information from StockTwits
        time_diff = datetime.datetime.now().timestamp() - file_time
        if time_diff > 86400 * 10:
            response = requests.get(self._stocktwits_api_symbol_infos_url)
            memory_file = io.StringIO(response.content.decode('utf-8'))
            symbol_infos = pd.read_csv(memory_file, header=None, names=["id", "symbol_name", "title",
                                                                        "industry", "sector"])
            # --- filter symbols by cashtag
            json_str = symbol_infos.to_json(orient='records')
            symbol_infos_json = json.loads(json_str)
            symbol_infos = [item for item in symbol_infos_json if item["symbol_name"].endswith(".X")]

            # --- write filtered symbols to a file in JSON format
            with open(self._symbols_info_filepath, "w", encoding="utf-8") as f:
                json.dump(symbol_infos, f, sort_keys=True, indent=4)
        else:
            symbol_infos = self.read_symbol_infos_from_file(filtered=False)
        return symbol_infos

    @staticmethod
    async def _get_mkt_cap(session, url):
        async with session.get(url) as resp:
            mkt_cap_text = ''
            resp_txt = await resp.text()
            crypto_info_txt = BS(resp_txt, "lxml")
            for tr in crypto_info_txt.find_all('tr')[1:]:
                tds = tr.find_all('td')
                row_name = tds[0].text.strip()
                if 'Market Cap' in row_name:
                    mkt_cap_text = tds[1].text
                    break
            if "B" in mkt_cap_text:
                return float(mkt_cap_text.replace('B', '')) * 1000000000
            else:
                return 0

    async def _retrieve_symbol_infos_filtered_by_mkt_cap_async_helper(self):
        # timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession() as session:
            tasks = list()
            for symbol_info in self._symbol_infos:
                symbol_ticker = symbol_info['symbol_name'].replace(".X", "") + '-USD'
                link_crypto = f'https://finance.yahoo.com/quote/{symbol_ticker}?p={symbol_ticker}'
                tasks.append(asyncio.ensure_future(self._get_mkt_cap(session, link_crypto)))
            tasks_gathered = await asyncio.gather(*tasks)
            return tasks_gathered

    def retrieve_symbol_infos_filtered_by_mkt_cap_async(self):
        file_time = 0
        # --- update the symbol list from stocktwits every 10 days
        if os.path.exists(self._filtered_symbols_info_filepath):
            file_time = os.path.getctime(self._filtered_symbols_info_filepath)
        time_diff = datetime.datetime.now().timestamp() - file_time
        if time_diff > 86400 * 10:
            loop = asyncio.get_event_loop()
            mkt_caps = loop.run_until_complete(self._retrieve_symbol_infos_filtered_by_mkt_cap_async_helper())
            loop.close()
            symbol_infos = list()
            for symbol_info, mkt_cap in zip(self._symbol_infos.copy(), mkt_caps):
                symbol_info['mkt_cap'] = mkt_cap
                symbol_infos.append(symbol_info)
            filtered_symbol_infos = [fsi for fsi in symbol_infos if fsi['mkt_cap'] > 0]
            # --- write filtered symbols to a file in JSON format
            with open(self._filtered_symbols_info_filepath, "w", encoding="utf-8") as f:
                json.dump(filtered_symbol_infos, f, sort_keys=True, indent=4)
        else:
            filtered_symbol_infos = self.read_symbol_infos_from_file()
        return filtered_symbol_infos

    def retrieve_symbol_infos_filtered_by_mkt_cap(self):
        s = requests.Session()
        filtered_symbol_infos = self._symbol_infos.copy()
        for symbol_info in filtered_symbol_infos:
            mkt_cap_number = 0
            mkt_cap_text = ''
            symbol_ticker = symbol_info['symbol_name'].replace(".X", "") + '-USD'
            link_crypto = f'https://finance.yahoo.com/quote/{symbol_ticker}?p={symbol_ticker}'
            crypto_info = s.get(link_crypto, timeout=5)
            crypto_info_txt = BS(crypto_info.text, "lxml")
            for tr in crypto_info_txt.find_all('tr')[1:]:
                tds = tr.find_all('td')
                row_name = tds[0].text.strip()
                if 'Market Cap' in row_name:
                    mkt_cap_text = tds[1].text
                    break
            if "B" in mkt_cap_text:
                mkt_cap_number = float(mkt_cap_text.replace('B', '')) * 1000000000
            else:
                pass
            symbol_info['mkt_cap'] = mkt_cap_number
        return [fsi for fsi in filtered_symbol_infos if fsi['mkt_cap'] > 0]

    def _dump_messages_to_database_helper(self, json_result, symbol_name):
        """Retrieve messages with max parameter from StockTwits and write results into database
           (if write_to_database = True)
           The returned value is the raw JSON format from StockTwits
        """
        # --- check if the HTTP request response is valid
        status = json_result["response"]["status"]

        if status == 200:  # Web Service Api Query OK
            json_result_ = json_result.copy()
            for msg in json_result_["messages"]:
                msg["querySymbolNames"] = [symbol_name]

            # --- store messages to database
            self._dump_messages_to_database(json_result_)
            return json_result_
        else:
            return json_result

    def retrieve_all_messages_from_stocktwits(self, symbol_name: str):
        """ Retrieves all messages from StockTwits for the given symbol_name.
        """
        print("\r\nRetrieving all messages for '{}'".format(symbol_name))
        retrieve_counter = 0
        while True:
            # --- get the minimum and maximum message ids from the database
            max_symbol_id = self.get_max_msg_id_for_symbol(symbol_name)
            print("\r\nTask[{0}] Retrieving {1} with max parameter {2}".format(retrieve_counter, symbol_name,
                                                                               max_symbol_id))

            # --- retrieve all messages for the given symbol and min/max ids
            retry_times = 0
            result = list()
            while retry_times < 5:
                result = self.retrieve_messages_from_stocktwits(symbol_name, max_symbol_id, True)

                if result == dict():
                    print("Retry times {0}, wait 5 sec for next retry".format(retry_times))
                    retry_times += 1
                    time.sleep(5)
                    continue
                else:
                    break
            if result == dict():
                break

            # --- get dates' info and update counters
            dates = [(msg["created_at"], msg["id"]) for msg in result["messages"]]
            if len(dates) == 0:
                break
            min_message_date_time = min(dates)
            max_message_date_time = max(dates)
            print("\r\nGet {0} messages [{1:%Y-%m-%d %H:%M:%S} (msg ID:{2}) ~ {3:%Y-%m-%d %H:%M:%S} (msg ID:{4})]".format(
                symbol_name, min_message_date_time[0], min_message_date_time[1], max_message_date_time[0],
                max_message_date_time[1]))

            # --- increase task counter
            retrieve_counter += 1
        return retrieve_counter

    def retrieve_messages_from_stocktwits(self, symbol_name, max_symbol_id: int, write_to_database: bool = True):
        """Retrieve messages with max parameter from StockTwits and write results into database
           (if write_to_database = True)
           The returned value is the raw JSON format from StockTwits
        """
        # --- read the latest messages from StockTwits Web Service
        response = self._api.get_stream(symbol_name, since=max_symbol_id)
        json_result = dict()

        # --- check if the HTTP request response is valid
        if response.ok:
            json_result = response.json()
            status = json_result["response"]["status"]

            if status == 200:  # Web Service Api Query OK
                for msg in json_result["messages"]:
                    msg["querySymbolNames"] = [symbol_name]

                # --- dump messages to database
                if write_to_database:
                    self._dump_messages_to_database(json_result)
        return json_result

    def read_symbol_infos_from_file(self, filtered: bool = True):
        """
        Load and return the coins' info
        """
        if filtered:
            file_path = self._filtered_symbols_info_filepath
        else:
            file_path = self._symbols_info_filepath
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def read_messages_from_database(self, symbol_id_or_name, start_dt: datetime, end_dt: datetime):
        raise NotImplementedError()

    @abstractmethod
    def _dump_messages_to_database(self, json_result):
        raise NotImplementedError()

    @abstractmethod
    def get_max_msg_id_for_symbol(self, symbol_id_or_name):
        raise NotImplementedError()

    @abstractmethod
    def get_min_max_msg_ids_for_all(self):
        raise NotImplementedError()
