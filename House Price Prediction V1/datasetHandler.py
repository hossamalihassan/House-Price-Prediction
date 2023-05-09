import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DatasetHandler:
    def __init__(self):
        self.data = pd.read_csv("dataset/train.csv")
        self.kaggle_test_data = pd.read_csv("dataset/test.csv")
        self.kaggle_Ids = np.array(self.kaggle_test_data["Id"])

    def print_dataset(self):
        print("Scaled down dataset : \n", self.data.head().to_string())
        print("Dataset shape : ", self.data.shape)

    def clean_dataset(self):
        # deal with categorical columns
        categorical_cols = ['Neighborhood']
        for col in categorical_cols:
            dummies = pd.get_dummies(self.data[col])
            self.data.join(dummies, how='left')

            test_dummies = pd.get_dummies(self.kaggle_test_data[col])
            self.kaggle_test_data.join(test_dummies, how='left')

        # feature eng (keep columns that has correlation with SalePrice between -0.2 and 0.2)
        important_cols = self.keep_important_cols(0.3)
        self.data = self.data[important_cols]
        self.save_dataset_metadata() # save before scaling down

        important_cols.remove("SalePrice")
        self.kaggle_test_data = self.kaggle_test_data[important_cols] # give test data the same columns of our data

        # scaling feature so it will be easier for gradient descent to converge
        self.data = self.scale_and_fill_null_features(self.data)
        self.kaggle_test_data = self.scale_and_fill_null_features(self.kaggle_test_data)

    def save_dataset_metadata(self):
        metadata = pd.DataFrame(columns=self.data.columns)
        for col in self.data.columns:
            metadata[col] = [self.data[col].min(), self.data[col].max(), self.data[col].mean()]

        metadata.to_csv("output/metadata.csv", index=False)

    def scale_and_fill_null_features(self, data):
        for col in data:
            data[col].fillna(data[col].mean(), inplace=True)
            data[col] = self.feature_scale_down(data[col])
        return data

    def keep_important_cols(self, value):
        important_num_cols = list(self.data.corr()["SalePrice"][(self.data.corr()["SalePrice"] > value) | (
                self.data.corr()["SalePrice"] < -value)].index)
        return important_num_cols

    def feature_scale_down(self, feature):
        return (feature - feature.min()) / (feature.max() - feature.min())

    def visualize_correlations(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr())
        plt.title("Correlations Between Variables", size=15)
        plt.show()

    def get_dataset(self):
        return (self.data, self.kaggle_test_data)

    def get_kaggle_Ids(self):
        return self.kaggle_Ids

    def feature_scale_up_y(self, y):
        y_price_scaled_up = []
        for i in range(len(y)):
            y_price_scaled_up.append(self.y_scale_up(y[i]))
        return y_price_scaled_up

    def y_scale_up(self, y):
        return round(
            y * round(self.data["SalePrice"].max() - self.data["SalePrice"].min()) + self.data["SalePrice"].min())
