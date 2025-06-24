"""
Example of exporting and importing energy consumption prediction model
"""

import pandas as pd
import sys
import os

sys.path.append('src')

from model import EnergyConsumptionPredictor
from data_processing import load_data, process_data

def main():
    df = load_data('data/energy_consumption.csv')
    df = process_data(df)

    model = EnergyConsumptionPredictor()
    model.train(df)

    saved_path = model.save_model('models/energy_model_latest.joblib')

    # Test loading
    loaded_model = EnergyConsumptionPredictor.from_file('models/energy_model_latest.joblib')

    # Test predictions - use absolute import for this case
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))

    predictions = loaded_model.predict_future(months=3)
    print(predictions[['Date', 'Predicted_Consumption', 'Predicted_Cost']].head())

    print(f"\nModel saved to: {saved_path}")

if __name__ == "__main__":
    main()
