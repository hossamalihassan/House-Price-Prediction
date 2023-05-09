from dataHandlerLSTM import DataHandlerLSTM
from modelPreprocessor import ModelPreprocessor
from modelLSTM import ModelLSTM
from predictLSTM import PredictLSTM
from updateDataset import UpdateDataset

class LSTM:

    def __init__(self):
        user_choice = self.take_user_choice()
        if(user_choice == 1):
            self.test_model()
        else:
            self.predict()

    def take_user_choice(self):
        print("--------------------")
        print("1 - Test the model")
        print("2 - Predict")
        choice = int(input("Your choice : "))
        return choice

    def test_model(self):
        data = self.handle_data()

        preprocessor = ModelPreprocessor(data)
        (X_train, X_test, y_train, y_test) = preprocessor.get_processed_data()

        self.run_model(X_train, X_test, y_train, y_test, preprocessor)

    def handle_data(self):
        handler = DataHandlerLSTM()
        return handler.get_data()

    def run_model(self, X_train, X_test, y_train, y_test, preprocessor):
        model = ModelLSTM()
        model.create_model(X_train, X_test, y_train, y_test)
        model.predict(preprocessor.get_scaler())
        score = model.get_model_score()
        print("--------------------")
        print("score = ", score)

        # save the model to use later when predicting
        model.save_model()

        # save updated dataset
        updateDataset = UpdateDataset(model.model)
        updateDataset.updateDataset()



    def predict(self):
        predictor = PredictLSTM()
        predictor.run()