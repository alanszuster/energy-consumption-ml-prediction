import pandas as pd
from src.data_processing import load_data, process_data
from src.visualization import plot_consumption, plot_costs, plot_predictions, display_monthly_summary
from src.model import train_model, predict_energy_consumption

data = load_data('data/energy_consumption.csv')
processed_data = process_data(data)

plot_consumption(processed_data)
plot_costs(processed_data)

poly, model = train_model(processed_data) 

future_dates = pd.date_range(start=processed_data['Date'].max() + pd.Timedelta(days=1), periods=12, freq='ME')
predictions = predict_energy_consumption(poly, model, future_dates)

display_monthly_summary(future_dates, predictions)
plot_predictions(future_dates, predictions)
