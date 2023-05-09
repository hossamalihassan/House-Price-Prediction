import pandas as pd

class DataHandlerLSTM:

    def __init__(self):
        self.dataset = self.read_data() # read the csv file first
        self.data = self.dataset
        self.handle_date_columns() # generate year and month columns from the data column

        self.data = self.data[['SUBURB', 'YEAR_SOLD', 'MONTH_SOLD', 'PRICE']] # only keep these columns to work on

        self.group_data_by_price() # group every month, year and suburb's mean price
        self.data = self.data.sort_values(by=["YEAR_SOLD", "MONTH_SOLD"]) # sort columns by year and month

        self.generate_suburb_dummies() # generate suburb dummies and append them to the dataset

    def read_data(self):
        return pd.read_csv("datasets/perth.csv")

    def print_data(self):
        print(self.data.head())

    def get_data(self):
        return self.data

    def get_og_data(self):
        return self.dataset

    def get_suburbs_dummies(self):
        return list(self.SUBURB_dummies.columns)

    def handle_date_columns(self):
        self.data["DATE_SOLD"] = pd.to_datetime(self.data["DATE_SOLD"])
        self.data['YEAR_SOLD'] = pd.DatetimeIndex(self.data['DATE_SOLD']).year
        self.data['MONTH_SOLD'] = pd.DatetimeIndex(self.data['DATE_SOLD']).month

    def group_data_by_price(self):
        self.data = self.data.groupby(['SUBURB', 'YEAR_SOLD', 'MONTH_SOLD'])['PRICE'].mean().reset_index()

    def generate_suburb_dummies(self):
        self.SUBURB_dummies = pd.get_dummies(self.data["SUBURB"])
        self.data = self.data.drop(["SUBURB"], axis=1)
        self.data = self.data.join(self.SUBURB_dummies)

    def group_by_suburb_price_mean(self):
        data_suburb_mean = self.dataset.groupby(["SUBURB"])["PRICE"].mean()
        return data_suburb_mean
