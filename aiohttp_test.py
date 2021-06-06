import aiohttp
import asyncio
from stwits_data_loader import StockTwitsClient

with open("stwits_data_loader/resources/access_token.txt", "r") as tokenFile:
    token = tokenFile.readline().rstrip()
api = StockTwitsClient()
api.login(token)


async def retrieve_async(symbol_name):
    async with aiohttp.ClientSession() as session:
        async with api.get_stream_async(symbol_name=symbol_name, session=session) as resp:
            result = await resp.json()
            print(result['messages'])

asyncio.run(retrieve_async("BTC.X"))




