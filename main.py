from src.data_processing import load_data, process_data
from src.visualization import plot_consumption, plot_costs, plot_predictions, display_monthly_summary, plot_model_performance
from src.model import EnergyConsumptionPredictor

def main():
    data = load_data('data/energy_consumption.csv')
    processed_data = process_data(data)

    print(f"Loaded {len(processed_data)} records")

    plot_consumption(processed_data)
    plot_costs(processed_data)

    predictor = EnergyConsumptionPredictor()
    performance = predictor.train(processed_data)

    print(f"Model trained - R²: {performance['r2_score']:.3f}")

    predictions_df = predictor.predict_future(months=12)

    display_monthly_summary(predictions_df)
    plot_predictions(predictions_df)
    plot_model_performance(predictor, processed_data, performance)

if __name__ == "__main__":
    main()
