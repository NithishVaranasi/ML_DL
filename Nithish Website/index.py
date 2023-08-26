from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    list = [int(x) for x in request.form.values()]

    loaded_model1 = pickle.load(open('DecisionTreeModel.pkl', 'rb'))
    loaded_model2 = pickle.load(open('AdaptiveBoostModel.pkl', 'rb'))
    loaded_model3 = pickle.load(open('KNNModel.pkl', 'rb'))
    final = np.array(list)
    prediction1 = loaded_model1.predict([final])
    prediction2 = loaded_model2.predict([final])
    prediction3 = loaded_model3.predict([final])
    predictions_list = [prediction1[0], prediction2[0], prediction3[0]]
    if predictions_list.count("Fraud") > predictions_list.count("No Fraud"):
        output = "Fraud Transaction"
    elif predictions_list.count("Fraud") < predictions_list.count("No Fraud"):
        output = "Not a Fraud Transaction"
    print(output)
    return render_template('index.html',pred = str(output))

if __name__ == "__main__":
    app.run(debug = True)
