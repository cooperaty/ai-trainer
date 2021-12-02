import pandas as pd
import pickle
import binance_client_setup


def recognize_cryptocurrencies_beginnings():
    # saves the date that reflect the beginning of every cryptocurrency pair
    # i.e, when began to be tradable in the market
    global beginnings

    # retrieves all available symbols (cryptocurrency pairs)
    # this way, we can query them to know more information about them
    exchange_info = binance_client_setup.client.get_exchange_info()
    available_symbols = [s["symbol"]
                         for s in exchange_info["symbols"] if "USDT" in s["symbol"]]

    # Binance began his operations on july 1st, 2017
    # and we want to retrieve batches of 1k candles
    # (candle size doesn't really matter)
    # this way, we will be getting, gradually, the very first cryptocurrencies
    # until now, including their operation dates (when they begin to be tradeable)
    beginning = pd.Timestamp(month=7, day=1, year=2017, unit="ms")
    candles_size = "1d"
    days_to_retrieve = 1000

    # will contain every cryptocurrency pair with their timestamps
    beginnings = {}

    # will retrieve existing information about cryptocurrencies beginnings
    # data can be found in "configuration/operations.pickle"
    # which is a backup to optimise execution times
    try:
        with open("configuration/operations.pickle", "rb") as f:
            beginnings = pickle.load(f)
    except FileNotFoundError:
        # first execution time should throw this error
        # so we skip it because we haven't created any file yet
        pass

    def flush(step: int, end: int):
        """
        Flush the console (only esthetical purposes).
        We want to do this before reaching the end,
        otherwise, we will get a blank output at the end
        """
        return print("\r", end="") if step != (end - 1) else print()

    for step, symbol in enumerate(available_symbols):
        # progress bar
        percentage = int(round((step+1)/len(available_symbols) * 100, 0))
        print("{step: >5}/{steps: <5} [{percentage}%] retrieving {symbol}".format(
            step=step+1, steps=len(available_symbols), percentage=percentage, symbol=symbol), end="")

        # if we do have information about the current cryptocurrency
        # we just simply skip it
        if symbol in beginnings:
            flush(step, len(available_symbols))
            continue

        # until we couldn't find the first record, we still querying Binance for the first record of the current symbol
        # consider that we query from the "beginning" so it is possible to not find out records of current symbol (mostly if it is new)
        while True:
            # retrieves candles from binance (it could be empty)
            bars = binance_client_setup.client.get_klines(symbol=symbol, interval=candles_size,
                                                          limit=days_to_retrieve, startTime=int(beginning.timestamp()))

            # if we get records from binance, we save the first record (they're sorted from oldest one and the last one)
            if len(bars) != 0:
                beginnings[symbol] = pd.Timestamp(bars[0][0], unit="ms")
                flush(step, len(available_symbols))
                break
            else:
                beginning = beginning + pd.DateOffset(days=days_to_retrieve)

    # saves beginning operation datetimes of every cryptocurrency pair
    with open("configuration/operations.pickle", "wb") as f:
        pickle.dump(beginnings, f)
