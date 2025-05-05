import pandas as pd
import logging
from src.data_processing import load_data, process_data
from src.visualization import plot_consumption, plot_costs, plot_predictions, display_monthly_summary
from src.model import train_model, predict_energy_consumption

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Rozpoczęcie przetwarzania danych")
        data = load_data('data/energy_consumption.csv')
        processed_data = process_data(data)
        
        plot_consumption(processed_data)
        plot_costs(processed_data)
        
        logger.info("Trenowanie modelu")
        poly, model, min_date = train_model(processed_data)   # odebranie min_date
        
        future_dates = pd.date_range(start=processed_data['Date'].max() + pd.Timedelta(days=1), periods=12, freq='ME')
        predictions = predict_energy_consumption(poly, model, future_dates, min_date)  # przekazanie min_date
        
        display_monthly_summary(future_dates, predictions)
        plot_predictions(future_dates, predictions)
        logger.info("Proces zakończony pomyślnie")
    except Exception as e:
        logger.error("Wystąpił błąd: %s", e)

if __name__ == "__main__":
    main()
