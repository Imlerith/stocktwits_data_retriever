from abc import ABC, abstractmethod
import os
import json
import time
import datetime
import requests
import io

import pandas as pd

from stwits_data_loader import StockTwitsClient


class DataLoader(ABC):
    """The base class which implements functionality to retrieve messages from StockTwits"""

    def __init__(self, token: str, min_msg_id: int = 200000000, max_total_id: int = 250000000):
        self._token = token
        self._min_msg_id = min_msg_id
        self._max_total_id = max_total_id
        self._stocktwits_api_symbol_infos_url = "http://api.stocktwits.com/symbol-sync/symbols.csv"
        self._api = StockTwitsClient()
        self._api.login(self._token)
        self._filtered_symbol_info_filename = os.getcwd() + "\\resources\\filteredSymbolInfos.json"

    def retrieve_symbol_infos_from_stocktwits(self, symbol_names_filter: list):
        """Retrive lastest symbolInfos from StockTwists, filtered by symbolNames
           the type of symbolNames is list
        """
        file_time = 0

        # --- update the symbol list from stocktwits every 10 days (???)
        if os.path.exists(self._filtered_symbol_info_filename):
            file_time = os.path.getctime(self._filtered_symbol_info_filename)

        # --- get the latest symbol information from StockTwits
        time_diff = datetime.datetime.now().timestamp() - file_time
        if time_diff > 86400 * 10:
            response = requests.get(self._stocktwits_api_symbol_infos_url)
            memory_file = io.StringIO(response.content.decode('utf-8'))
            symbol_infos = pd.read_csv(memory_file, header=None, names=["id", "symbol_name", "title",
                                                                        "industry", "sector"])
            # --- filter symbols
            filtered_symbol_infos = symbol_infos.loc[symbol_infos["symbol_name"].isin(symbol_names_filter)]\
                .to_dict("records")

            # --- write filtered symbols to a file with JSON format
            with open(self._filtered_symbol_info_filename, "w", encoding="utf-8") as f:
                json.dump(filtered_symbol_infos, f, sort_keys=True, indent=4)
        else:
            filtered_symbol_infos = self.read_symbol_infos_from_file()
        return filtered_symbol_infos

    def retrieve_symbol_with_dot_x_infos_from_stocktwits(self):
        """Retrive lastest symbolInfos from StockTwists, filtered by symbolNames
           the type of symbolNames is list
        """
        file_time = 0

        # --- update the symbol list from stocktwits every 10 days
        if os.path.exists(self._filtered_symbol_info_filename):
            file_time = os.path.getctime(self._filtered_symbol_info_filename)

        # --- get the latest symbol information from StockTwits
        time_diff = datetime.datetime.now().timestamp() - file_time
        if time_diff > 86400 * 10:
            response = requests.get(self._stocktwits_api_symbol_infos_url)
            memory_file = io.StringIO(response.content.decode('utf-8'))
            symbol_infos = pd.read_csv(memory_file, header=None, names=["id", "symbol_name", "title",
                                                                        "industry", "sector"])
            # --- filter symbols
            json_str = symbol_infos.to_json(orient='records')
            symbol_infos_json = json.loads(json_str)
            filtered_symbol_infos = [item for item in symbol_infos_json if item["symbol_name"].endswith(".X")]

            # --- write filtered symbols to a file with JSON format
            with open(self._filtered_symbol_info_filename, "w", encoding="utf-8") as f:
                json.dump(filtered_symbol_infos, f, sort_keys=True, indent=4)
        else:
            filtered_symbol_infos = self.read_symbol_infos_from_file()

        return filtered_symbol_infos

    def retrieve_all_messages_from_stocktwits(self, symbol_name: str, wait_counter: int = 2):
        """ Retrieves all messages from StockTwits for the given symbol_name between min_message_id
            and max_message_id (using 'since' and 'max' parameters of the API). Both the min and
            max ids are incremented in a loop.
        """
        retrieve_counter = 0
        while True:
            # --- get the minimum and maximum message ids from the database
            _, max_message_id = self._get_min_max_msgs_ids_for_symbol(symbol_name)
            print("\r\nTask[{0}] Retrieving {1} with max parameter {2}".format(retrieve_counter, symbol_name,
                                                                               max_message_id))

            # --- retrieve all messages for the given symbol and min/max ids
            retry_times = 0
            result = list()
            while retry_times < 5:
                result, _ = self.retrieve_messages_from_stocktwits(symbol_name, max_message_id, True)

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

    def retrieve_messages_from_stocktwits(self, symbol_name, max_symbol_id: int, write_to_database: bool = True):
        """Retrieve messages with max parameter from StockTwits and write results into database
           (if write_to_database = True)
           The returned value is the raw JSON format from StockTwits
        """
        # --- read the latest messages from StockTwits Web Service
        response = self._api.get_stream(symbol_name, since=max_symbol_id, max=self._max_total_id)
        at_least_one_msg_recorded = False
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
                    at_least_one_msg_recorded = self._dump_messages_to_database(json_result)
        return json_result, at_least_one_msg_recorded

    def set_min_msg_id(self, min_id):
        self._min_msg_id = min_id

    def set_max_total_id(self, max_id):
        self._max_total_id = max_id

    @staticmethod
    def read_symbol_infos_from_file():
        """Return the list of currencies in database.
        """
        file_path = os.getcwd() + "\\resources\\filteredSymbolInfos.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def read_messages_from_database(self, symbol_id_or_name, start_dt: datetime, end_dt: datetime):
        raise NotImplementedError()

    @abstractmethod
    def _dump_messages_to_database(self, json_result):
        raise NotImplementedError()

    @abstractmethod
    def _get_min_max_msgs_ids_for_symbol(self, symbol_id_or_name):
        raise NotImplementedError()

    @abstractmethod
    def get_min_max_msg_ids_for_all(self):
        raise NotImplementedError()
