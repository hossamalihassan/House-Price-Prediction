import numpy as np
import pickle
from dataHandlerLR import DataHandlerLR
from makingPredictions import MakingPredictions

class PredictLR:
    def __init__(self):
        self.dataHandlerLR = DataHandlerLR()
        self.dataset = self.dataHandlerLR.get_dataset()
        self.suburb_dummies = self.dataHandlerLR.get_suburbs_dummies()
        self.load_model()

        self.take_user_inputs()

    def run(self):
        makingPredictions = MakingPredictions(self.model, self.suburb_dummies, self.dataset)
        self.suburb_index = self.get_suburb_index()
        prediction_input = makingPredictions.create_prediction_array(self.suburb_index, self.inputs)

        prediction_input = np.reshape(prediction_input, (prediction_input.shape[2]))

        prediction = makingPredictions.make_prediction(prediction_input)
        print("Predicted price = ", prediction)

    def take_user_inputs(self):
        self.suburb_input = input("Suburb : ").lower()
        if (self.check_for_suburb(self.suburb_input)):
            self.inputs = self.take_features_input()
        else:
            self.take_user_inputs()

    def take_features_input(self):
        inputs = []
        features = self.dataHandlerLR.get_features_without_suburbs()
        features = np.delete(features, 0)
        for feature in features:
            inp = input(feature + ": ")
            inputs.append(inp)
        return inputs

    def check_for_suburb(self, suburb_input):
        self.suburb_dummies = self.convert_suburbs_to_lowercase(self.suburb_dummies)
        if suburb_input not in self.suburb_dummies:
            print('Invalid suburb !')
            print("1 - Choose from the suburbs list")
            print("2 - Try again")
            suburb_user_choice = int(input("Your choice : "))
            if suburb_user_choice == 1:
                self.show_suburbs_list(self.suburb_dummies)
                self.suburb_input = input("Your choice : ")
            else:
                return False

        return True

    def show_suburbs_list(self, suburbs):
        for i in range(len(suburbs)):
            print(i + 1, " - ", suburbs[i])

    def convert_suburbs_to_lowercase(self, suburbs):
        return [x.lower() for x in suburbs]

    def get_suburb_index(self):
        if (isinstance(self.suburb_input, int)):
            suburb_index = self.suburb_input - 1
        else:
            suburb_index = np.where(np.array(self.suburb_dummies) == self.suburb_input)[0][0]

        return suburb_index

    def load_model(self):
        with open('model/lr_model.pickle', 'rb') as f:
            self.model = pickle.load(f)