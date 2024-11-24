# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from model import load_data, train_model, predict_price

app = Flask(__name__)

# Load the CSV file and train the model
df = load_data('your_stock_data.csv')  # Make sure the path is correct
model = train_model(df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form submitted by the user
    date = request.form['date']
    open_price = float(request.form['open_price'])
    high_price = float(request.form['high_price'])
    low_price = float(request.form['low_price'])
    volume = float(request.form['volume'])

    # Convert date to numerical format (ordinal)
    date = pd.to_datetime(date).toordinal()

    # Prepare input data
    input_data = np.array([date, open_price, high_price, low_price, volume])

    # Make the prediction
    predicted_price = predict_price(model, input_data)
    
    return render_template('index.html', prediction=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
