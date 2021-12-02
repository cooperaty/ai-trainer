import error
import os
from binance.client import Client


def initialize_binance_client():
    # defines the client to make requests to the Binance API
    global client

    # gets Binance API credentials
    # those are defined in environment variables named, respectively:
    # - binance_api_key: the api key obtained from Binance
    # - binance_api_secret: the api secret obtained from Binance
    api_key = os.environ.get("binance_api_key")
    api_secret = os.environ.get("binance_api_secret")

    # if neither of the environment variables from above are defined, the program stops
    if not api_key:
        raise error.NonDefinedEnvironmentVariableError("binance_api_key")
    if not api_secret:
        raise error.NonDefinedEnvironmentVariableError("binance_api_secret")

    # if any of the environment variables is empty, the program stops
    if api_key == "":
        raise error.EmptyEnviromentVariableError("binance_api_key")
    if api_secret == "":
        raise error.EmptyEnviromentVariableError("binance_api_secret")

    client = Client(api_key, api_secret)
