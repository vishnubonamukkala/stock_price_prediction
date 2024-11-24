# model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_model(df):
    # Convert 'Date' to numerical values for the model
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

    # Define features (X) and target variable (y)
    X = df[['Date', 'Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model's performance
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    
    return model

def predict_price(model, input_data):
    return model.predict([input_data])
