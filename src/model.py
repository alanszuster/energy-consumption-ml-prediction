from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(data, degree=3):

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    
    X = data[['Days']]
    y = data['Consumption']
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return poly, model

def predict_energy_consumption(poly, model, future_dates):
  
    future_days = (future_dates - future_dates.min()).days
    future_days = future_days.values.reshape(-1, 1)
    future_days_poly = poly.transform(future_days)
    predictions = model.predict(future_days_poly)

    return predictions
