import numpy as np
from datasetHandler import DatasetHandler
import pandas as pd

class Model:

    def __init__(self):
        self.weights = []
        self.Y_hat = []
        self.dataHandler = DatasetHandler()

    def split_data_into_test_and_train(self, data, train_portion):
        train_data_size = int(len(data) * (train_portion))
        self.test_data = data.iloc[train_data_size:, :]
        self.train_data = data.iloc[:train_data_size, :]

        print("train data -> ", len(self.train_data), " rows")
        print("test data  -> ", len(self.test_data), "  rows")

        return (self.train_data, self.test_data)

    def calc_y_minus_y_mean(self, Y):
        Y_mean = Y.mean()
        Y_minus_Y_mean = []
        Y_minus_Y_mean_sq_sum = 0
        for i in range(len(Y)):
            Y_minus_Y_mean.append(Y.iloc[i] - Y_mean)
            Y_minus_Y_mean_sq = pow(Y_minus_Y_mean[i], 2)
            Y_minus_Y_mean_sq_sum += Y_minus_Y_mean_sq
        return (Y_minus_Y_mean, Y_minus_Y_mean_sq_sum)

    def calc_x_minus_x_mean(self, X, X_len, i, Xi_mean, Y_minus_Y_mean):
        X_minus_X_mean_sq_sum = 0
        X_Y_prod_sum = 0
        for j in range(X_len):
            X_minus_X_mean = X.iloc[j][i] - Xi_mean
            X_minus_X_mean_sq = pow(X_minus_X_mean, 2)
            X_minus_X_mean_sq_sum += X_minus_X_mean_sq
            X_Y_prod_sum += X_minus_X_mean * Y_minus_Y_mean[j]

        return (X_minus_X_mean_sq_sum, X_Y_prod_sum)

    def calc_X_mean_for_each_X(self, X):
        X_mean_arr = []
        for i in range(len(X.columns)):
            X_mean_arr.append(np.mean(X.iloc[:, i]))
        return X_mean_arr

    def calc_weight_for_each_feature(self, X_Y_prod_sum, Y_minus_Y_mean_sq_sum, X_minus_X_mean_sq_sum, Y_len, X_len):
        R = self.calc_R(X_Y_prod_sum, Y_minus_Y_mean_sq_sum, X_minus_X_mean_sq_sum)
        Sy = self.calc_Sx_and_Sy(Y_minus_Y_mean_sq_sum, Y_len)
        Sx = self.calc_Sx_and_Sy(X_minus_X_mean_sq_sum, X_len)
        weight = R * (Sy / Sx)

        return weight

    def calc_R(self, X_Y_prod_sum, Y_minus_Y_mean_sq_sum, X_minus_X_mean_sq_sum):
        return X_Y_prod_sum / np.sqrt(Y_minus_Y_mean_sq_sum * X_minus_X_mean_sq_sum)

    def calc_Sx_and_Sy(self, sq_sum, length):
        return np.sqrt(sq_sum / (length - 1))

    def calc_intercept(self, X_len, X_mean_arr):
        intercept = self.weights[0]
        for i in range(1, X_len):
            intercept -= self.weights[i] * X_mean_arr[i - 1]
        self.weights[0] = intercept

    def train(self, train_data):
        X = train_data.drop(["SalePrice"], axis=1)
        Y_price = train_data["SalePrice"]
        X_len = len(X.columns)
        Y_len = len(Y_price)

        # calc y - mean(y) and the sum of pow(y - mean(y), 2)
        (Y_minus_Y_mean, Y_minus_Y_mean_sq_sum) = self.calc_y_minus_y_mean(Y_price)

        # calc x_mean for each feature
        X_mean_arr = self.calc_X_mean_for_each_X(X)

        self.weights.append(Y_price.mean())
        for i in range(X_len):
            Xi_mean = X_mean_arr[i]

            # calc X_minus_X_mean_sq_sum and the sum of X_Y_prod
            (X_minus_X_mean_sq_sum, X_Y_prod_sum) = self.calc_x_minus_x_mean(X, X_len, i, Xi_mean, Y_minus_Y_mean)

            # calc weight
            weight = self.calc_weight_for_each_feature(X_Y_prod_sum, Y_minus_Y_mean_sq_sum, X_minus_X_mean_sq_sum, Y_len, X_len)
            self.weights.append(weight)

        # calc intercept
        self.calc_intercept(X_len, X_mean_arr)

    def calc_y_hat_minus_y_prod_x(self, x, i, y):
        y_hat_minus_y_prod_x_sum = 0
        for j in range(len(x)):
            y_hat_minus_y = self.Y_hat[j] - y.iloc[j]
            y_hat_minus_y_prod_x = y_hat_minus_y * x.iloc[j][i]
            y_hat_minus_y_prod_x_sum += y_hat_minus_y_prod_x
        return y_hat_minus_y_prod_x_sum

    def calc_cost(self, y):
        Y_resd = y - self.Y_hat
        cost = np.sum(np.dot(Y_resd.T, Y_resd)) / len(y - Y_resd)
        return cost

    def scale_up_y_and_y_hat(self, y):
        y_scaled_up = self.dataHandler.feature_scale_up_y(y)
        return y_scaled_up

    def scale_up(self, column, type):
        if type == "y_hat":
            scaled_up = self.scale_up_y_and_y_hat(column)
        else:
            scaled_up = self.scale_up_y_and_y_hat(column.tolist())

        return scaled_up

    def calc_r2(self, y):
        y_mean = np.mean(np.array(y))
        SSR = np.sum(np.power(np.array(y) - np.array(self.Y_hat), 2))
        SST = np.sum(np.power(np.array(y) - y_mean, 2))
        r2 = (1 - (SSR / SST))

        return r2

    def modify_weights(self, x, y, learning_rate):
        Y_hat_diff_Y_sum = np.sum(y - self.Y_hat)
        self.weights[0] -= (1 * (Y_hat_diff_Y_sum / len(x)) * learning_rate)
        for i in range(len(x.columns)):
            y_hat_minus_y_prod_x = self.calc_y_hat_minus_y_prod_x(x, i, y)
            self.weights[i + 1] -= (y_hat_minus_y_prod_x) * (learning_rate / len(x))

    def grad_desc(self, x, y, learning_rate, max_iteration):
        learning_rate = learning_rate
        max_iteration = max_iteration

        min_cost = 1
        for itr in range(max_iteration):
            # modify weights
            self.modify_weights(x, y, learning_rate)

            # calc y_hat after modifying the weights
            self.Y_hat = self.calc_Y_hat(x)

            # calc cost
            cost = self.calc_cost(y)
            print("Iteration = ", itr + 1, "\tCost = ", cost)

            # stop when we reach the lowest cost
            if (cost < min_cost):
                min_cost = cost
            else:
                break

        self.save_weights_to_csv()

    def predict(self, X):
        predictions = self.calc_Y_hat(X)

        self.scale_up_y_and_y_hat(predictions)
        y_hat_price_scaled_up = self.dataHandler.feature_scale_up_y(predictions)
        kaggle_ids = self.dataHandler.get_kaggle_Ids()
        output_df = pd.DataFrame({"Id": kaggle_ids, "SalePrice": y_hat_price_scaled_up})
        output_df.to_csv("output/output.csv", index=False)

    def calc_Y_hat(self, X):
        Y_hat_arr = []
        for i in range(len(X)):
            y_hat = 0
            for j in range(len(X.columns)):
                if (j == 0):
                    y_hat = self.weights[j]
                else:
                    y_hat += float(X.iloc[i][j] * self.weights[j])

            Y_hat_arr.append(y_hat)

        return Y_hat_arr

    def evaluate(self, test):
        # scale up y_hat and y
        Y = test["SalePrice"]
        X = test.drop(["SalePrice"], axis=1)

        self.Y_hat = self.calc_Y_hat(X)
        y_hat_scaled_up = self.scale_up(self.Y_hat, "y_hat")
        y_scaled_up = self.scale_up(Y, "y")

        y_y_hat_df = pd.DataFrame(
            {"y": y_scaled_up, "y_hat": y_hat_scaled_up, "diff": np.array(y_scaled_up) - np.array(y_hat_scaled_up)})
        print("Y, Y_hat and the diff : ")
        print(y_y_hat_df.head(40))

        # calc r2
        r2 = self.calc_r2(Y)
        print("R-squared = ", r2)

    def fit(self):
        # calc y_hat before doing gd
        Y_price = self.train_data["SalePrice"]
        X = self.train_data.drop(["SalePrice"], axis=1)
        self.Y_hat = self.calc_Y_hat(X)

        # do gd to fit the data
        self.grad_desc(X, Y_price, 0.1, 100)

    def save_weights_to_csv(self):
        weights_df = pd.DataFrame({"weight" : self.weights})
        weights_df.to_csv("output/weights.csv", index=False)