import keras.layers
import numpy as np

class ModelLSTM:

    def __init__(self):
        self.model = keras.models.Sequential()

    def create_model(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        self.create_layers() # first, we create the model layers
        self.compile_model() # we compile the model
        self.fit_model(50, 5) # fit the model

    def create_layers(self):
        self.model.add(keras.layers.LSTM(50, activation="relu", return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(keras.layers.LSTM(50, activation="relu", return_sequences=False))
        self.model.add(keras.layers.Dense(25))
        self.model.add(keras.layers.Dense(1))

    def compile_model(self):
        self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    def fit_model(self, batch, epochs):
        print("--------------------")
        print("Model epochs: ")
        self.model.fit(self.X_train, self.y_train, batch_size=batch, epochs=epochs)

    def predict(self, scaler):
        print("--------------------")
        print("Making Predictions: ")
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)

        self.test_predictions = scaler.inverse_transform(self.test_predictions)
        self.train_predictions = scaler.inverse_transform(self.train_predictions)

        self.y_train = scaler.inverse_transform(self.y_train)
        self.y_test = scaler.inverse_transform(self.y_test)

    def get_model_score(self):
        return self.calc_r2(self.y_test, self.test_predictions) * 100

    def calc_r2(self, y, y_hat):
        y_mean = np.mean(np.array(y))
        SSR = np.sum(np.power(np.array(y) - np.array(y_hat), 2))
        SST = np.sum(np.power(np.array(y) - y_mean, 2))
        r2 = (1 - (SSR / SST))

        return r2

    def save_model(self):
        model_json = self.model.to_json()
        with open("model/lstm_model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model/lstm_model.h5")




