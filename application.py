import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template,request,jsonify

application = Flask(__name__)
app=application
ridge_model=pickle.load(open("models/ridge.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))
@app.route('/')
def index():
    return render_template("home.html")
@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))

        data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, Classes]])

        data_scaled = standard_scaler.transform(data)
        prediction = ridge_model.predict(data_scaled)

        return render_template('result.html', results=prediction[0])
    else:
        return render_template("home.html")
if __name__ == '__main__':
    app.run(host="0.0.0.0")
