import pandas as pd

def convert_date(date_str):
    return pd.to_datetime(date_str, format='%d.%m.%Y')
