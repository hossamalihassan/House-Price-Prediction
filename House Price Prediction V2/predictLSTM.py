import keras
import numpy as np
from dataHandlerLSTM import DataHandlerLSTM
from makingPredictions import MakingPredictions

class PredictLSTM:

    def __init__(self):
        self.data_handler = DataHandlerLSTM()
        self.data = self.data_handler.get_data()
        self.suburb_dummies = self.data_handler.get_suburbs_dummies()

    def run(self):
        self.load_model() # load saved model

        self.take_user_inputs()

        makingPredictions = MakingPredictions(self.model, self.suburb_dummies, self.data)
        self.suburb_index = self.get_suburb_index()
        prediction_input = makingPredictions.create_prediction_array(self.suburb_index, [self.year_input, self.month_input])

        prediction = makingPredictions.make_prediction(prediction_input)
        print("Predicted price = ", prediction)

    def take_user_inputs(self):
        self.suburb_input = input("Suburb : ").lower()
        if(self.check_for_suburb(self.suburb_input)):
            self.year_input = input("Year : ")
            self.month_input = input("Month : ")
        else:
            self.take_user_inputs()

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
            print(i+1 , " - " , suburbs[i])

    def convert_suburbs_to_lowercase(self, suburbs):
        return [x.lower() for x in suburbs]

    def get_suburb_index(self):
        if(isinstance(self.suburb_input, int)):
            suburb_index = self.suburb_input - 1
        else:
            suburb_index = np.where(np.array(self.suburb_dummies) == self.suburb_input)[0][0]

        return suburb_index

    def load_model(self):
        json_file = open('model/lstm_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model/lstm_model.h5")