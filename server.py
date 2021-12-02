from flask import Flask
from flask import request
from flask_cors import CORS

import numpy as np
import pickle
import binance_client_setup
import cryptocurrencies_setup
import exercises_builder

# the number of local exercise to train
exercises_amount = 3

# the number of candles per exercise,
# those will be the points you will see in the plot
total_candles = 1000

# total indicators to send in the response
# that is, Open, High, Low, Close, Volume,
# or well-known as OHLCV
total_indicators = 6

# initializes the binance client to make request to the Binance API
# the client indeed is a global variable named "client"
binance_client_setup.initialize_binance_client()

# recognizes every cryptocurrencies beginnings, that is,
# when they became tradable (in terms of the datetime)
# the result is stored into a global dictionary named "beginnings" where:
# - keys are the name of the cryptocurrency pair
# - values are the datetime (pandas.Timestamp type) of the beginning of the cryptocurrency
# also the dictionary is persisted into a file named "operations.pickle" (located in "./configuration")
cryptocurrencies_setup.recognize_cryptocurrencies_beginnings()

# builds local exercises to give to the users
# results are stored into a file named "x_training.pickle" and "y_training.pickle" where:
# - x_training.pickle is a numpy array of dimension N x 6
# - y_training.pickle is a numpy array of dimension N x 1
# both cases, N is the total number of exercises
# but 6 is the number of indicators, i.e, OHLCV (Open, High, Low, Close, Volume)
# and 1 the trending, i.e, -1 for downward trending, 0 for range and 1 for upward trending
exercises_builder.build_exercises(
    exercises_amount=exercises_amount, candles_amount=total_candles)

# defines the flask app and allows CORS in local development
# but, once the code is in production, CORS(app) must be removed for security
app = Flask(__name__)
CORS(app)

# loads training data
with open("training/x_training.pickle", "rb") as f:
    x_training = pickle.load(f)
    x_training = np.array(x_training).reshape(-1,
                                              total_candles, total_indicators)
with open("training/y_training.pickle", "rb") as f:
    y_training = pickle.load(f)
    y_training = np.array(y_training).reshape(-1, 1)

    # there is a typo in the dataset where classes begin at value -1 to 1
    # instead of 0 to 2, this way, we fix it up by adding one unit
    # this will be removed in the future
    y_training += 1


@app.route("/exercise")
def exercise():
    """
    Generates a new exercise to practice trading
    """
    user_hash = request.args.get("user_hash")

    if not user_hash:
        return {"error": "You forget to include the user_hash key"}

    total_exercises = x_training.shape[0]
    random_index = np.random.randint(0, total_exercises)

    random_x = x_training[random_index].tolist()
    random_y = y_training[random_index].tolist()

    return {"x_training": random_x, "y_training": random_y, "exercise_hash": random_index}
