from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

class ModelLR:

    def __init__(self):
        self.model = LinearRegression()

    def create_model(self, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        self.fit_model() # fit the model

    def fit_model(self):
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[2]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[2]))
        self.model.fit(self.X_train, self.y_train)

    def predict(self, scaler):
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)

        self.test_predictions = scaler.inverse_transform(self.test_predictions)
        self.train_predictions = scaler.inverse_transform(self.train_predictions)

        self.y_train = scaler.inverse_transform(self.y_train)
        self.y_test = scaler.inverse_transform(self.y_test)

    def get_model_score(self):
        return self.calc_r2(self.test_predictions, self.y_test) * 100

    def calc_r2(self, y, y_hat):
        y_mean = np.mean(np.array(y))
        SSR = np.sum(np.power(np.array(y) - np.array(y_hat), 2))
        SST = np.sum(np.power(np.array(y) - y_mean, 2))
        r2 = (1 - (SSR / SST))

        return r2

    def save_model(self):
        with open('model/lr_model.pickle', 'wb') as f:
            pickle.dump(self.model, f)