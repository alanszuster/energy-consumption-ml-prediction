import pandas as pd
import numpy as np

# Polish energy rates (2024)
ENERGY_RATE = 0.5050
DISTRIBUTION_MULTIPLIER = 0.817
VAT_RATE = 0.23

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    return df

def process_data(df):
    df = df.dropna().sort_values(by='Date')

    df['Reading'] = df['Reading'].astype(str).str.replace(' kWh', '', regex=False).astype(float)

    # Calculate consumption from readings
    if 'Consumption' in df.columns and df['Consumption'].sum() > 0:
        df = df[df['Consumption'] > 0]
    else:
        df['Consumption'] = df['Reading'].diff().fillna(0)
        df = df[df['Consumption'] >= 0]

    # Calculate costs if not provided
    if 'Cost' not in df.columns or df['Cost'].sum() <= 0:
        energy_cost = df['Consumption'] * ENERGY_RATE
        distribution_fee = energy_cost * DISTRIBUTION_MULTIPLIER
        subtotal = energy_cost + distribution_fee
        vat = subtotal * VAT_RATE
        df['Cost'] = subtotal + vat

    # Basic date features
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days

    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    return df
