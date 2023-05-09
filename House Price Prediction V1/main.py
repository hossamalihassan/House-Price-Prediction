from datasetHandler import DatasetHandler
from model import Model
from predictionApp import App

def test_model():
    # cleaning the dataset and getting it ready
    dataHandler = DatasetHandler()
    dataHandler.clean_dataset()
    # dataHandler.visualize_correlations()
    dataHandler.print_dataset()

    (train_data, kaggle_test_data) = dataHandler.get_dataset()

    model = Model()
    (train_data, test_data) = model.split_data_into_test_and_train(train_data, train_portion=0.80)

    model.train(train_data)
    model.fit()
    model.evaluate(test_data)
    model.predict(kaggle_test_data)

def prediction_app():
    app = App()
    app.take_inputs()
    app.predict()

def main():
    print("-----------------------")
    print("House prices prediction")
    print("1 - Predict a house price")
    print("2 - Test the model")
    user_choice = int(input("Enter your choice: "))
    print("-----------------------")
    if(user_choice == 1):
        prediction_app()
    elif(user_choice == 2):
        test_model()

main()

