import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# Starting Flash app
app = Flask(__name__)
# Loading Pickle file
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output)) 

# To Start running the Flash app
if __name__ == "__main__":
    app.run(debug=True)
    

