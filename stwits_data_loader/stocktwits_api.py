import requests
from aiohttp import ClientSession


class StockTwitsClient:
    """Class for accessing StockTwits API functions.

    Methods:
      login -- Change the token used for login.
      search -- Search for users, symbols, or both.
      stream_symbol -- Download the stream of a symbol.

    For more detail on StockTwits API, see the documentation at
    https://api.stocktwits.com/developers/docs/api
    """

    def __init__(self):
        self.base = "https://api.stocktwits.com/api/2/"  # Base URL of all requests.
        self.token = None
        self.login("")  # By default, your API client is unregistered.

    def login(self, token):
        """Change the token used for login.

        If token is an empty str, StockTwits API will see you as an unregistered
        user and rate limits will be lower.
        """
        self.token = token

    def search(self, mode=None, **kwargs):
        """Search for users, symbols, or both.

        You should call search with the keyword argument q, where q is the str you
        want to search (e.g. self.search(mode="symbols", q="BTC")).
        mode can have value None, "users" or "symbols". In the first case,
        the API will return a mix of users and symbols."""
        url = self.base + "search{}.json".format("" if mode is None else "/" + mode)
        if self.token != "":
            kwargs.update({"access_token": self.token})
        result = requests.get(url, params=kwargs)
        return result

    def get_stream(self, symbol_name, **kwargs):
        """Download the stream of the symbol identified by given symbol name.

        The result will contain at most 30 messages.
        Additional arguments such as since and max can be used to select a specific
        time period. See StockTwits documentation for more detail.

        The getParams includes:
            id (symbol_name): Ticker symbol, Stock ID, or RIC code of the symbol (Required)
            since:	Returns results with an ID greater than (more recent than) the specified ID.
            max:	Returns results with an ID less than (older than) or equal to the specified ID.
            limit:	Default and max limit is 30. This limit must be a number under 30.
            callback:	Define your own callback function name, add this parameter as the value.
            filter:	Filter messages by links, charts, videos, or top. (Optional)
        """
        url = self.base + "streams/symbol/{}.json".format(symbol_name)

        if self.token != "":
            kwargs.update({"access_token": self.token})
        try:
            result = requests.get(url, params=kwargs)
        except Exception as e:
            print("HTTP Request fail {}\r\n{}".format(url, e))
            return None

        return result

    def get_stream_async(self, symbol_name: str, session: ClientSession, **kwargs):
        url = self.base + "streams/symbol/{}.json".format(symbol_name)

        if self.token != "":
            kwargs.update({"access_token": self.token})
        try:
            result = session.get(url, params=kwargs)
        except Exception as e:
            print("HTTP Request fail {}\r\n{}".format(url, e))
            return None

        return result



