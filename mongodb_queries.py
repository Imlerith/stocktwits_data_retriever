import datetime
import pymongo

db_name = "stocktwits_msgs"
host = "127.0.0.1"
port = 27017
username = ""
password = ""


client = pymongo.MongoClient(host, port)
database = client[db_name]
collection = database["Messages"]

query = {
    "querySymbolNames": 'ETH.X',
    "created_at": {"$gt": datetime.datetime(2020, 1, 1), "$lt": datetime.datetime(2020, 1, 2)}
}

query_result = collection.find(query)
query_result_list = list(query_result)
collection.distinct('symbols.symbol')

coll_agg = collection.aggregate([
        {
            "$unwind": "$symbols"
        },
        {
            "$unwind": "$symbols.symbol"
        },
        { "$group": {"_id": {}, "uniqueValues": { "$addToSet": "$symbols.symbol"}} }
    ])

