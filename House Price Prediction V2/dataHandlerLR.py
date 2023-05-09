import pandas as pd
import datetime
import numpy as np

class DataHandlerLR:
    def __init__(self):
        self.dataset = self.read_dataset()
        self.get_data_ready()

        self.selected_features = self.get_selected_features()
        self.features_without_suburbs = self.selected_features
        self.selected_features = self.merge_suburbs_with_selected_features()
        self.dataset = self.dataset[self.selected_features]

    def read_dataset(self):
        year = str(datetime.datetime.now().year)
        month = str(datetime.datetime.now().month)
        src = "datasets/perth-updated-" + year + "-" + month + ".csv"
        return pd.read_csv(src)

    def get_data_ready(self):
        # create suburb dummies
        SUBURB_dummies = pd.get_dummies(self.dataset["SUBURB"])
        self.suburbs = list(SUBURB_dummies.columns)
        self.dataset = self.dataset.join(SUBURB_dummies)

        # remove columns
        self.dataset = self.dataset.drop(["ADDRESS", "SUBURB", "NEAREST_SCH", "NEAREST_STN", "DATE_SOLD", "NEAREST_SCH"],axis=1)

    def get_selected_features(self):
        corr_matrix = self.dataset.corr()
        selected_features = corr_matrix[corr_matrix['PRICE'] > 0.2].index.tolist()
        return selected_features

    def get_features_without_suburbs(self):
        return self.features_without_suburbs

    def merge_suburbs_with_selected_features(self):
        return np.insert(self.suburbs, 0, self.selected_features)

    def get_dataset(self):
        return self.dataset

    def get_suburbs_dummies(self):
        return self.suburbs
