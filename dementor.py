from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.models import load_model


class Dementor:
    def __init__(self, model_path=None):
        if model_path:
            self.model = load_model(model_path)
            return

        closings_shape = 1000
        tendence_shape = 500
        user_choices = 3  # downward, range, upward

        decisions_input = Input(1)
        decisions_embedding = Embedding(
            input_dim=user_choices, output_dim=tendence_shape)(decisions_input)
        decisions_flatten = Flatten()(decisions_embedding)

        closings_input = Input(closings_shape)
        closings_dense = Dense(64)(closings_input)
        closings_relu = LeakyReLU(alpha=0.2)(closings_dense)
        closings_dropout = Dropout(rate=0.3)(closings_relu)

        closings_dense = Dense(128)(closings_dropout)
        closings_relu = LeakyReLU(alpha=0.2)(closings_dense)
        closings_dropout = Dropout(rate=0.3)(closings_relu)

        decisions_closings_concatenation = Concatenate(
            axis=1)([decisions_flatten, closings_dropout])
        decisions_closings_dense = Dense(64)(decisions_closings_concatenation)
        decisions_closings_relu = LeakyReLU(
            alpha=0.2)(decisions_closings_dense)
        decision_closings_dropout = Dropout(rate=0.2)(decisions_closings_relu)

        decisions_closings_dense = Dense(128)(decisions_closings_concatenation)
        decisions_closings_relu = LeakyReLU(
            alpha=0.2)(decisions_closings_dense)
        decisions_closings_dropout = Dropout(rate=0.2)(decisions_closings_relu)

        decisions_closings_dense = Dense(256)(decision_closings_dropout)
        decisions_closings_relu = LeakyReLU(
            alpha=0.2)(decisions_closings_dense)
        decisions_closings_dropout = Dropout(rate=0.2)(decisions_closings_relu)

        output = Dense(user_choices, activation="softmax")(
            decisions_closings_dropout)
        model = Model([decisions_input, closings_input], output)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam", metrics=["accuracy"])

        self.model = model

    def train_on_batch(self, inputs, outputs):
        metrics = self.model.train_on_batch(inputs, outputs, return_dict=True)
        return metrics

    def save_model(self, name):
        self.model.save(f"models/{name}.h5", include_optimizer=True)

    def evaluate(self, inputs, outputs):
        metrics = self.model.evaluate(inputs, outputs, return_dict=True)
        return metrics
