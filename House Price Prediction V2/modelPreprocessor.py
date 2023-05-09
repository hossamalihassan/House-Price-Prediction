import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class ModelPreprocessor:

    def __init__(self, data):
        self.data = data
        (self.X_train, self.X_test, self.y_train, self.y_test) = self.split_data_into_train_test() # split the dataset into two portions

        self.scale_down() # scale down X and Y for the model

        # reshape the inputs
        self.X_train = self.reshape_x(self.X_train)
        self.X_test = self.reshape_x(self.X_test)

    def generate_x_y_from_data(self):
        X = self.data.drop(["PRICE"], axis=1)
        y = self.data["PRICE"]
        return (X, y)

    def split_data_into_train_test(self):
        (X, y) = self.generate_x_y_from_data()  # get the model inputs (X) and predicted output (y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        # reshape  y
        y_train = self.reshape_y(y_train)
        y_test = self.reshape_y(y_test)

        return (X_train, X_test, y_train, y_test)

    def reshape_y(self, y):
        return np.reshape(y, (y.shape[0], 1))

    def scale_down(self):
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.fit_transform(self.X_test)
        self.y_train = self.scaler.fit_transform(self.y_train)
        self.y_test = self.scaler.fit_transform(self.y_test)

    def get_scaler(self):
        return self.scaler

    def reshape_x(self, X):
        return np.reshape(X, (X.shape[0], 1, X.shape[1]))

    def get_processed_data(self):
        return (self.X_train, self.X_test, self.y_train, self.y_test)