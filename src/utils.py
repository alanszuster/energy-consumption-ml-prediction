import pandas as pd

def convert_date(date_str):
    return pd.to_datetime(date_str, format='%d.%m.%Y')

def calculate_statistics(df):
    return {
        'total_consumption': df['Consumption'].sum(),
        'average_monthly': df['Consumption'].mean(),
        'total_cost': df['Cost'].sum(),
        'cost_per_kwh': df['Cost'].sum() / df['Consumption'].sum(),
        'data_points': len(df)
    }
