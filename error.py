import pandas


class NonDefinedEnvironmentVariableError(Exception):
    """
    Raises when the environment variable is not defined (used to retrieve Binance API secret keys)
    """

    def __init__(self, environment_variable_name: str):
        """
        Parameters
        ----------
        environment_variable_name: str
            The name of the environment variable
        """
        super().__init__(
            f"The environment variable {environment_variable_name} is not defined")


class EmptyEnviromentVariableError(Exception):
    """
    Raises when the environment variable is an empty string
    """

    def __init__(self, environment_variable_name: str):
        """
        Parameters
        ----------
        environment_variable_name: str
            The name of the environment variable
        """
        super().__init__(
            f"The environment variable {environment_variable_name} is empty")


class BeforeOperationError(Exception):
    """
    Raises when requires data from a cryptocurrency which began after the time indicated in the argument
    """

    def __init__(self, symbol: str, beginning: pandas.Timestamp, start: pandas.Timestamp):
        """
        Parameters
        ----------
        symbol: str
            The name of the cryptocurrency
        beginning: pandas.Timestamp
            The datetime when the cryptocurrency begun their operations (to be tradable)
        start: pandas.Timestamp
            The starting datetime to retrieve data from the cryptocurrency
        """
        super().__init__(
            f"{symbol} began their operations on {beginning}, but the function requires data from {start} which is before then")


class NotEnoughCandlesError(Exception):
    """
    Raises when there are not enough candles to retrieve considering the starting time indicated in the argument
    """

    def __init__(self, symbol: str, candles_size: str, start: pandas.Timestamp):
        """
        Parameters
        ----------
        symbol: str
            The name of the cryptocurrency
        candle_size: str
            The size of the candles, either, 15m, 30m or 1h, for instance
        start: pandas.Timestamp
            The starting datetime to retrieve data from the cryptocurrency
        """
        super().__init__(
            f"{symbol} doesn't have enough candles of size {candles_size} to retrieve from {start}")


class NotEnoughCandlesFromBinanceError(Exception):
    """
    Raises when Binance returns a number of candles less than the indicated in the argument.
    Theoretically this should never happen, but there are "holes" in the historical data
    """

    def __init__(self, candles_size: str, symbol: str):
        """
        Parameters
        ----------
        symbol: str
            The name of the cryptocurrency
        candle_size: str
            The size of the candles, either, 15m, 30m or 1h, for instance
        """
        super().__init__(
            f"Binance didn't get enough candles of size {candles_size} for {symbol}, even when it should theoretically (maybe missing data on Binance database)")


class BinanceMaxCandlesError(Exception):
    """
    Raises when requires more candles than binance can give
    """

    max_candles_to_retrieve = 1000

    def __init__(self, candles_provided: int):
        """
        Parameters
        ----------
        candles_provided: int
            The number of candles to retrieve from Binance API
        """
        super().__init__(
            f"Binance can only retrieve {BinanceMaxCandlesError.max_candles_to_retrieve} candles, but you provided {candles_provided} which is greater")
