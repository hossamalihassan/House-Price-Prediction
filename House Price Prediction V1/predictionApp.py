import pandas as pd

class App:
    def __init__(self):
        metadata_df = pd.read_csv("output/metadata.csv")
        self.sale_price_metadata = metadata_df["SalePrice"].to_list()
        self.metadata = metadata_df.drop(["SalePrice"], axis=1)

        weights_df = pd.read_csv("output/weights.csv")
        self.weights = weights_df["weight"].to_list()

    def take_inputs(self):
        self.inputs = []
        for col in self.metadata:
            input_prompt = col + " (from " + str(self.metadata[col][0]) + " to " + str(self.metadata[col][1]) + ") : "
            self.inputs.append(float(input(input_prompt)))

        self.check_inputs_for_outliers()

    def check_inputs_for_outliers(self):
        # if the input is outlier replace it with the mean of the column
        for i in range(len(self.inputs)):
            if(self.metadata.iloc[0][i] > self.inputs[i] or self.metadata.iloc[1][i] < self.inputs[i]):
                self.inputs[i] = self.metadata.iloc[2][i]

    def predict(self):
        self.inputs = self.scale_down_inputs(self.inputs)
        prediction = 0
        for i in range(len(self.inputs)):
            if (i == 0):
                prediction = self.weights[i]
            else:
                prediction += float(self.inputs[i] * self.weights[i])

        prediction = self.scale_up_price(prediction)

        print("-----------------------")
        print("Predicted price = ", prediction)

    def scale_down_inputs(self, inputs):
        for i in range(len(inputs)):
            inputs[i] = (inputs[i] - self.metadata.iloc[0][i]) / (self.metadata.iloc[1][i] - self.metadata.iloc[0][i])
        return inputs

    def scale_up_price(self, price):
        return round(price * round(self.sale_price_metadata[1] - self.sale_price_metadata[0]) + self.sale_price_metadata[0])