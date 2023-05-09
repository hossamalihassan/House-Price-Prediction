import pandas as pd
import numpy as np

class DataHandlerLR:
    def __init__(self):
        self.dataset_metadata = self.read_dataset("perth-metadata.csv")
        suburbs_dataset = self.read_dataset("suburbs.csv")
        self.suburbs = suburbs_dataset.values
        
    def get_dataset_metadata(self):
        return self.dataset_metadata

    def get_suburbs_list(self):
        return list(self.suburbs)

    def get_features(self):
        features = list(self.dataset_metadata.columns)
        features.remove("PRICE")
        return features

    def read_dataset(self, path):
        src = "dataset/" + path
        return pd.read_csv(src)