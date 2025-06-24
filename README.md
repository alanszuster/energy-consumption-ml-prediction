# Energy Consumption Prediction

Predicts household energy consumption using machine learning.

🤗 **Pre-trained model available on Hugging Face:** [alanoee/energy_prediction_model](https://huggingface.co/alanoee/energy_prediction_model)

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## What it does

- Loads energy meter readings from CSV
- Trains ML models (Random Forest, Gradient Boosting, Linear Regression)
- Predicts next 12 months consumption and costs
- Creates charts and analysis

## Data format

CSV with columns: `Date`, `Reading` (meter reading in kWh)

## Output files

- `energy_consumption_analysis.png` - Historical trends
- `energy_predictions.png` - 12-month forecast
- `model_performance_analysis.png` - Model comparison

## Notebooks

- `01_data_exploration.ipynb` - Data analysis
- `02_model_development.ipynb` - Model testing
- `03_predictions.ipynb` - Forecasting

## Pre-trained Model

Want to use the model without training? Download the pre-trained model from [Hugging Face](https://huggingface.co/alanoee/energy_prediction_model):

```bash
git clone https://huggingface.co/alanoee/energy_prediction_model
cd energy_prediction_model
pip install -r requirements.txt
python example.py
```
