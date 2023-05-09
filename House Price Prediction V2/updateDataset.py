import datetime
import pandas as pd
import numpy as np
from dataHandlerLSTM import DataHandlerLSTM
from makingPredictions import MakingPredictions

class UpdateDataset:

    def __init__(self, model):
        data_handler = DataHandlerLSTM()
        data = data_handler.get_data()
        suburb_dummies = data_handler.get_suburbs_dummies()

        self.makingPredictions = MakingPredictions(model, suburb_dummies, data)

        self.suburbs = self.makingPredictions.suburb_dummies
        og_date = data_handler.get_og_data()
        self.suburb_mean = og_date.groupby(["SUBURB"])["PRICE"].mean()
        self.predictions_list = []

        self.dataHandlerLSTM = DataHandlerLSTM()
        self.dataset = self.dataHandlerLSTM.dataset

    def updateDataset(self):
        print("\-- Updating the dataset --/")

        self.predictions_list = self.get_predicted_mean_price_for_each_suburb() # first we calculate the difference between the predicted price and the old price
        self.merge_dataset_with_diff_in_mean()
        self.add_the_diff_in_mean_to_the_old_price()
        self.dataset = self.dataset.drop(["DIFF_PRICE", "SUBURBS"], axis=1)
        src = "datasets/perth-updated-" + str(datetime.datetime.now().year) + "-" + str(datetime.datetime.now().month) + ".csv"
        self.dataset.to_csv(src)

        print("--- Dataset is updated ---")

    def get_predicted_mean_price_for_each_suburb(self):
        # get mean house prices in each suburb in the current time
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        predictions_list = []
        for i in range(len(self.makingPredictions.suburb_dummies)):
            prediction_input = self.makingPredictions.create_prediction_array(i,  [year, month])
            prediction = self.makingPredictions.make_prediction(prediction_input)
            predictions_list.append(prediction)

        return predictions_list


    def get_diff_between_price_and_predicted_price_dataframe(self):
        diff_in_mean = {"SUBURBS": self.suburbs, "DIFF_PRICE": (self.predictions_list - self.suburb_mean)}
        diff_in_mean_df = pd.DataFrame(diff_in_mean).reset_index()
        return diff_in_mean_df

    def merge_dataset_with_diff_in_mean(self):
        diff_in_mean_df = self.get_diff_between_price_and_predicted_price_dataframe()
        self.dataset = pd.merge(self.dataset, diff_in_mean_df, on="SUBURB")

    def add_the_diff_in_mean_to_the_old_price(self):
        self.dataset["PRICE"] = self.dataset["PRICE"] + self.dataset["DIFF_PRICE"]
