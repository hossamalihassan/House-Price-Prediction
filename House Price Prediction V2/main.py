from LSTM import LSTM
from LinearReg import LinearReg

class App:

    def __init__(self):
        user_choice = self.take_user_choice()
        if(user_choice == 1):
            lstm = LSTM()
        else:
            linearReg = LinearReg()

    def take_user_choice(self):
        print("1 - Predict house price (based on date and suburb)")
        print("2 - Estimate house price (based on the house features) ** an up-to-date dataset **")
        choice = int(input("Your choice : "))
        return choice

app = App()