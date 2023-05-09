from flask import Flask, render_template, url_for, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from dataHandlerLR import DataHandlerLR
from predictLR import PredictLR

app = Flask(__name__)
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
db = SQLAlchemy(app)

class Recent_Predictions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    suburb = db.Column(db.String(200))
    bedrooms = db.Column(db.Integer, default=0)
    bathrooms = db.Column(db.Integer, default=0)
    floor_area = db.Column(db.Integer, default=0)
    price = db.Column(db.Integer, default=0)

    def __repr__(self) -> str:
        return '<h1 %r>' % self.id

@app.route('/', methods=['POST', 'GET'])
def index():
    dataHandler = DataHandlerLR()
    features = dataHandler.get_features()
    suburbs = dataHandler.get_suburbs_list()
    if request.method == "POST":
        inputs_list = [request.form["suburb"]]
        for feature in features:
            inputs_list.append(request.form[feature])
        predict = PredictLR(inputs_list)
        prediction = predict.get_prediction_result()

        recent_prediction = Recent_Predictions(suburb=inputs_list[0], bedrooms=inputs_list[1], bathrooms=inputs_list[2], floor_area=inputs_list[3], price=prediction)
        try:
            db.session.add(recent_prediction)
            db.session.commit()
        except:
            return "something went wrong when trying to insert the prediction into the database"

        prediction = format(prediction, ",")
        recent_predictions = Recent_Predictions.query.order_by(Recent_Predictions.id.desc()).limit(5).all();
        return render_template('index.html', suburbs=suburbs, features=features, inputs_list=inputs_list, prediction=prediction, recent_predictions=recent_predictions)
    else:
        recent_predictions = Recent_Predictions.query.limit(5).all();
        return render_template('index.html', suburbs=suburbs, features=features, inputs_list=[], recent_predictions=recent_predictions)

if __name__ == "__main__":
    app.run(debug=True)