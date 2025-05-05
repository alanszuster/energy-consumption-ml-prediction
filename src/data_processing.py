import pandas as pd
import logging
from pandas import DataFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENERGY_RATE = 0.5050
DISTRIBUTION_MULTIPLIER = 0.817
VAT_RATE = 0.23

def load_data(file_path: str) -> DataFrame:
    """
    Wczytuje dane z pliku CSV z kolumną 'Date'.

    :param file_path: Ścieżka do pliku CSV.
    :return: DataFrame z wczytanymi danymi.
    """
    logger.info("Wczytywanie danych z %s", file_path)
    df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    return df

def process_data(df: DataFrame) -> DataFrame:
    """
    Przetwarza dane:
    - Usuwa wiersze z brakującymi danymi.
    - Sortuje dane wg daty.
    - Oblicza zużycie energii oraz koszt jako suma opłaty za energię, opłaty dystrybucyjnej i VAT.

    :param df: Surowe dane.
    :return: Przetworzony DataFrame.
    """
    df = df.dropna()
    df = df.sort_values(by='Date')

    # Konwersja kolumny "Reading": najpierw rzutujemy na string a następnie usuwamy " kWh" i przekształcamy na float
    df['Reading'] = df['Reading'].astype(str).str.replace(' kWh', '', regex=False).astype(float)

    df['Consumption'] = df['Reading'].diff().fillna(0)
    df = df[df['Consumption'] >= 0]
    
    energy_cost = df['Consumption'] * ENERGY_RATE
    distribution_fee = energy_cost * DISTRIBUTION_MULTIPLIER
    subtotal = energy_cost + distribution_fee
    vat = subtotal * VAT_RATE
    df['Cost'] = subtotal + vat

    return df
