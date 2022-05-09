from operator import mod
import pandas as pd
from flask import Flask, jsonify, request
import joblib
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    input = req['data']
    df = pd.DataFrame.from_dict(input)

    model = joblib.load('model.pkl')

    pred = model.predict(df)

    if pred == 0 :
        typ = 'setosa'
    elif pred == 1:
        typ = 'versicolor'
    else:
        typ = 'virginica'

    return jsonify({'output':typ})

def home():
    return "iris data set sample model"

if __name__=='__main__':
    port = os.environ.get("PORT", 9999)
    app.run(debug=False,host='0.0.0.0', port=port)