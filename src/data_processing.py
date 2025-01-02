import pandas as pd

COST_PER_KWH = 0.75

def load_data(file_path):
    
    df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    return df

def process_data(df):

    df = df.dropna()
    df = df.sort_values(by='Date')

    df['Consumption'] = df['Reading'].diff().fillna(0)
    df = df[df['Consumption'] >= 0]
    df['Cost'] = df['Consumption'] * COST_PER_KWH

    return df
