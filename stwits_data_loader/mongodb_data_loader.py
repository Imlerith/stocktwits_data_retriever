import datetime

import pymongo

from stwits_data_loader import DataLoader


class MongoDBDataLoader(DataLoader):

    def __init__(self, db_name, host: str = "127.0.0.1", port: int = 27017, username: str = "", password: str = "",
                 *args, **kwargs):
        # --- establish MongoDB database connection
        super().__init__(*args, **kwargs)
        self._dbName = db_name
        self._client = pymongo.MongoClient(host, port)
        self._database = self._client[db_name]
        self._collection = self._database["Messages"]

        index_name = "id"
        if index_name not in self._collection.index_information():
            self._collection.create_index(index_name, unique=True)

        if username != "" or password != "":
            self._database.authenticate(username, password)

    def __del__(self):
        # --- close and release database resources
        self._client.close()

    def _dump_messages_to_database(self, json_result):
        """
        Writing messages in JSON format to database
        :param json_result: message in JSON fomat
        :return: nothing, write messages to database
        """
        collection = self._collection
        # --- record messages into database if not already there
        for msg_dict in json_result["messages"]:

            # --- convert datetime string into datetime. This allows filtering documents by time operators in MongoDB
            msg_dict["created_at"] = datetime.datetime.strptime(msg_dict["created_at"], "%Y-%m-%dT%H:%M:%SZ")\
                .replace(tzinfo=datetime.timezone.utc)
            try:
                collection.insert_one(msg_dict)
                print('+', end='')
            except pymongo.errors.DuplicateKeyError:  # the message already exists in the database
                print('.', end='')

                # --- update querySymbolNames
                query = {'id': msg_dict['id']}
                doc = collection.find_one(query)
                if msg_dict['querySymbolNames'][0] not in doc['querySymbolNames']:
                    query_symbol_name = doc['querySymbolNames'] + msg_dict['querySymbolNames']
                    collection.update_one({'id': msg_dict['id']},
                                          {'$set': {'querySymbolNames': query_symbol_name}})
            except Exception as e:
                print("Insert Error: {}", e)

    def read_messages_from_database(self, symbol_id_or_name="", start_dt: datetime = datetime.datetime(2020, 1, 1),
                                    end_dt: datetime = datetime.datetime.now()):
        """Read messages from MongoDB
           The returned messages have an additional key-value pair '_id'.
           The key-value pair is given by MongoDB automatically.
           '_id' is the primary key for each 'document' in the MongoDB database.
        """
        cursor = self._read_message_by_pymongo_cursor(symbol_id_or_name, start_dt, end_dt)
        if cursor is None:
            return list()
        else:
            # --- convert cursor to list of dict
            return list(cursor)

    def _read_message_by_pymongo_cursor(self, symbol_id_or_name="",
                                        start_dt: datetime = datetime.datetime(2020, 1, 1),
                                        end_dt: datetime = datetime.datetime.now()):

        if type(symbol_id_or_name) is int:
            raise Exception("Query by symbolId not supported")
        if type(symbol_id_or_name) is not str:
            return None

        collection = self._collection
        if symbol_id_or_name == "":
            query = {"created_at": {"$gt": start_dt, "$lt": end_dt}}
        else:
            query = {
                "querySymbolNames": symbol_id_or_name,
                "created_at": {"$gt": start_dt, "$lt": end_dt}
            }
        return collection.find(query)

    def get_max_msg_id_for_symbol(self, symbol_id_or_name):

        pipeline = [{'$match': {'querySymbolNames': symbol_id_or_name}},
                    {'$group': {'_id': {}, 'max': {'$max': '$id'}}}]

        cursor = self._collection.aggregate(pipeline)
        value = list(cursor)
        if not value:
            max_id = self._min_msg_id
        else:
            max_id = value[0]['max']
        return max_id

    def get_min_max_msg_ids_for_all(self):
        pipeline = [{'$group': {'_id': {}, 'min': {'$min': '$id'}, 'max': {'$max': '$id'}}}]
        cursor = self._collection.aggregate(pipeline)
        value = list(cursor)
        if not value:
            return None, None
        else:
            return value[0]['min'], value[0]['max']


