from dataHandlerLR import DataHandlerLR
from predictLR import PredictLR
from modelPreprocessor import ModelPreprocessor
from modelLR import ModelLR

class LinearReg:

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
        dataHandlerLR = DataHandlerLR()
        data = dataHandlerLR.get_dataset()
        preprocessor = ModelPreprocessor(data)
        (X_train, X_test, y_train, y_test) = preprocessor.get_processed_data()

        self.run_model(X_train, X_test, y_train, y_test, preprocessor)

    def run_model(self, X_train, X_test, y_train, y_test, preprocessor):
        model = ModelLR()
        model.create_model(X_train, X_test, y_train, y_test)
        model.predict(preprocessor.get_scaler())
        score = model.get_model_score()
        print("--------------------")
        print("score = ", score)

        # save the model to use later when predicting
        model.save_model()

    def predict(self):
        predictLR = PredictLR()
        predictLR.run()