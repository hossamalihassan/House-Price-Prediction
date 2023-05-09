import numpy as np

class MakingPredictions:
    def __init__(self, model, suburb_dummies, metadata):
        self.model = model
        self.suburb_dummies = suburb_dummies
        self.metadata = metadata

    def create_prediction_array(self, suburb_index, features_input):
        # merge dummies suburb array with year and month
        prediction_dummies = self.create_suburb_dummies_array(suburb_index)
        prediction_input = np.insert(prediction_dummies, 0, features_input)

        # reshape and scale down
        prediction_input = np.reshape(prediction_input, (1, prediction_input.shape[0]))
        prediction_input = self.scale_down(prediction_input)

        # reshape input to predict
        return np.reshape(prediction_input, (1, 1, prediction_input.shape[1]))

    def make_prediction(self, prediction_input):
        prediction = self.model.predict([prediction_input])
        prediction = self.scale_price_up(prediction[0][0])
        return prediction

    def create_suburb_dummies_array(self, suburb_index):
        prediction_dummies = np.zeros((len(self.suburb_dummies),))
        prediction_dummies[suburb_index] = 1
        return prediction_dummies

    def scale_down(self, arr):
        i = 0
        for col in self.metadata.columns:
            if col != "PRICE":
                arr[0][i] = (arr[0][i] - self.metadata.iloc[0][col]) / (self.metadata.iloc[1][col] - self.metadata.iloc[0][col])
                i += 1

        return arr

    def scale_price_up(self, price):
        return round(price * round(self.metadata.iloc[1]["PRICE"] - self.metadata.iloc[0]["PRICE"]) + self.metadata.iloc[0]["PRICE"])