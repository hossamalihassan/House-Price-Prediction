import numpy as np
import pickle
from dataHandlerLR import DataHandlerLR
from makingPredictions import MakingPredictions

class PredictLR:
    def __init__(self, inputs_list):
        self.inputs_list = inputs_list
        self.dataHandlerLR = DataHandlerLR()
        self.dataset_metadata = self.dataHandlerLR.get_dataset_metadata()
        self.suburb_dummies = self.dataHandlerLR.get_suburbs_list()

        self.load_model()
        self.run()

    def run(self):
        makingPredictions = MakingPredictions(self.model, self.suburb_dummies, self.dataset_metadata)
        self.suburb_index = self.get_suburb_index()
        prediction_input = makingPredictions.create_prediction_array(self.suburb_index, self.inputs_list[1:])
        prediction_input = np.reshape(prediction_input, (prediction_input.shape[2]))
        self.prediction = makingPredictions.make_prediction(prediction_input)

    def get_suburb_index(self):
        return np.where(np.array(self.suburb_dummies) == self.inputs_list[0])[0][0]

    def get_prediction_result(self):
        return self.prediction

    def load_model(self):
        with open('model/lr_model.pickle', 'rb') as f:
            self.model = pickle.load(f)