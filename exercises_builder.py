import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.linear_model import LinearRegression
from error import BinanceMaxCandlesError, BeforeOperationError, NotEnoughCandlesError, NotEnoughCandlesFromBinanceError
import cryptocurrencies_setup
import binance_client_setup
from utils import flush

# when we want to retrieve a finite number of candles from Binance API, let's say N,
# we are restricted to:
# - the datetime we want to start getting data
# - the candle size, for instance, 15m, 1h, 4h, and so on
# then we need to use a base unit (this case minutes) to predict if the combination of arguments
# allow us to retrieve N candles considering the starting datetime and candle sizes
# so, this variable help us to cast every candle size into minutes,
# this way, if we want to get 10 candles of "1h" size
# we know that "1h" is equivalent to 60 minutes, and then,
# we will require 10 * 60 = 600 minutes minimum
# and the starting datetime should be, at least,
# 600 minutes backwards of the moment we made the request
converter = {"15m": 1/4 * 60, "1h": 1 * 60, "4h": 4 * 60}


def get_candles(symbol: str, start: pd.Timestamp, candles_amount: int, candles_size: str):
    """
    Retrieves candles of a cryptocurrency pair from Binance API

    Parameters
    ----------
    symbol: str
        The cryptocurrency pair to retrieve
    start: pandas.Timestamp
        The datetime to start retrieving candles
    candles_amount: int
        Amount of candles to retrieve
    candles_size: str
        The size of the candles to retrieve, either, 15m, 30m, and so on.

    Raises
    ------
    error.BeforeOperationError
        If the starting datetime to request candles is before the existence of the cryptocurrency pair 
    error.NotEnoughCandlesError
        If the combination of starting datetime to retrieve candles and the candle size itself doesn't allow to retrieve the indicated number of candles
    error.NotEnoughCandlesFromBinanceError
        If retrieved candles from Binance API are less than expected (even having the right arguments)
    """
    # if candles_amount > BinanceMaxCandlesError.max_candles_to_retrieve:
    #     raise BinanceMaxCandlesError(candles_amount)

    today = pd.Timestamp.today()
    symbol_beginning = cryptocurrencies_setup.beginnings[symbol]

    if start < symbol_beginning:
        raise BeforeOperationError(
            symbol=symbol, beginning=cryptocurrencies_setup.beginnings[symbol], start=start)

    elapsed = (today - start)
    candles = int(
        np.floor(elapsed / pd.Timedelta(converter[candles_size], "m")))

    if candles < candles_amount:
        raise NotEnoughCandlesError(
            symbol=symbol, candles_size=candles_size, start=start)

    candles_left = candles_amount
    bars = np.empty((0, 12), float)

    while candles_left:
        candles_to_retrieve = candles_left

        if candles_left > BinanceMaxCandlesError.max_candles_to_retrieve:
            candles_to_retrieve = BinanceMaxCandlesError.max_candles_to_retrieve

        retrieved_bars = binance_client_setup.client.get_klines(
            symbol=symbol, interval=candles_size, startTime=int(start.timestamp()), limit=candles_to_retrieve)
        bars = np.append(bars, retrieved_bars, axis=0)
        candles_left -= candles_to_retrieve

    if len(bars) < candles_amount:
        raise NotEnoughCandlesFromBinanceError(
            candles_size=candles_size, symbol=symbol)

    return bars.tolist()


def trending(closings: np.array):
    """
    Recognizes the trending of a cryptocurrency pair market

    Parameters
    ----------
    closings: numpy.array
        2D dimensional array where rows are 
    """

    # because we don't have the independent variable
    # (we only have the outputs those are the closings)
    # we must make it
    # this way, x=0 maps to y=closings[0]
    # then x=1 maps to y=closings[1], and so on
    timestamps = np.arange(closings.shape[0]).reshape(-1, 1)
    regression = LinearRegression().fit(timestamps, closings)

    # once regression is done, we standarize the slope
    # this way, values will be independent of output magnitudes
    slope = regression.coef_[0]
    timestamps_standard_deviation = np.std(timestamps)
    closings_standard_deviation = np.std(closings)
    standarized_slope = slope * timestamps_standard_deviation / \
        closings_standard_deviation

    # upward trend when standarized_slope >= 0.5 returns 1
    # downward trend when standarized_slope <= -0.5 returns -1
    # neither upward nor downward trend returns 0
    return 1 * (standarized_slope >= 0.5) + -1 * (standarized_slope <= -0.5), regression


def get_exercises(exercises_amount: int, candles_amount: int = 512, candles_size: str = "15m"):
    """
    Generates exercises retrieving candles from Binance API to predict the tendence
    """

    # gets all available symbols from Binance
    symbols = list(cryptocurrencies_setup.beginnings.keys())
    x_training = []
    y_training = []

    exercises_created = 0

    while exercises_created < exercises_amount:
        # choose random symbol to workout
        random_index = torch.randint(
            low=0, high=len(symbols), size=(1,))[0].item()
        symbol = symbols[random_index]

        # if we want to get 200 candles of 15m starting from today, we couldn't
        # because we can't get data that doesn't exist in Binance yet
        # this way, we must request the candles, at least, 200 * 15 minutes = 3000 minutes before today
        # now, we could request candles from the beginning of the cryptocurrency until today minus 3000 minutes
        # in general, the range would be [today, today - candles to retrieve * candle_frame]
        # in this case, we use minutes as basic unit time
        # that's the reason why we use a converter because if we have a candle frame like 1h
        # it will be converted as 60 (minutes).
        today = pd.Timestamp.today()
        limit_datetime = today - \
            pd.DateOffset(minutes=converter[candles_size]*candles_amount)
        minutes_range = np.ceil(pd.Timedelta(
            limit_datetime - cryptocurrencies_setup.beginnings[symbol]).total_seconds() / 60).astype(int)

        # minutes range could be negative
        # it means that the cryptocurrency is too new
        # then, there are not enough data to retrieve
        if minutes_range < 0:
            continue

        # we choose a random minute between [0, minutes_range]
        # this will add up minutes from the beginning of operations of the cryptocurrency
        # this way, we will choose random dates depending on how much minutes we want to be far away from the beginning
        random_minutes = torch.randint(
            low=0, high=minutes_range, size=(1,))[0].item()
        start = cryptocurrencies_setup.beginnings[symbol] + \
            pd.DateOffset(minutes=random_minutes)

        # request candles from Binance
        try:
            # progress bar
            percentage = int(round((exercises_created+1) /
                             exercises_amount * 100, 0))
            print("{step: >5}/{steps: <5} [{percentage}%] making exercise with {symbol}".format(
                step=exercises_created+1, steps=exercises_amount, percentage=percentage, symbol=symbol), end="")

            # retrieved bars from Binance API
            bars = get_candles(symbol=symbol, start=start,
                               candles_amount=candles_amount, candles_size=candles_size)
            bars = np.array(bars).astype(float)
            closings = bars[:, 4]

            # consider the following indexes
            # bars[i][1] = open
            # bars[i][2] = high
            # bars[i][3] = low
            # bars[i][4] = close
            # bars[i][5] = volume
            # with that, we flatten the data and order will keep
            # but with the following indexes
            # x_training[i][j][0] = open
            # x_training[i][j][1] = high
            # x_training[i][j][2] = low
            # x_training[i][j][3] = close
            # x_training[i][j][4] = volume
            trend, _ = trending(closings)
            x_training.append(bars[:,:6])
            y_training.append(trend)

            # clear the output to print in the same line
            # otherwise (reaching the end) prints out a new line
            flush(exercises_created, exercises_amount)

            # increase the amount of exercises created
            exercises_created += 1

        except NotEnoughCandlesFromBinanceError:
            print(
                f"[ERROR] candle size: {candles_size} candles_amount: {candles_amount} symbol: {symbol} start: {start}")

    return x_training, y_training


def build_exercises(exercises_amount, candles_amount):
    # gets training data
    x_training, y_training = get_exercises(
        exercises_amount=exercises_amount, candles_amount=candles_amount)

    # persist data into pickles
    with open("training/x_training.pickle", "wb") as f:
        pickle.dump(x_training, f)
    with open("training/y_training.pickle", "wb") as f:
        pickle.dump(y_training, f)
