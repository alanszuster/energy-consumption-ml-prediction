from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import logging
from pandas import DataFrame
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data: DataFrame, degree: int = 3):
    """
    Trenuje model regresji wielomianowej.

    :param data: Przetworzone dane z kolumną 'Date' i 'Consumption'.
    :param degree: Stopień wielomianu.
    :return: Krotka (poly, model, min_date)
    """
    logger.info("Trenowanie modelu z wielomianem stopnia %d", degree)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    min_date = data['Date'].min()               # dodano: zapamiętanie minimalnej daty
    data['Days'] = (data['Date'] - min_date).dt.days

    X = data[['Days']]
    y = data['Consumption']

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    return poly, model, min_date               # zwracamy min_date

def predict_energy_consumption(poly, model, future_dates: pd.Series, min_date: pd.Timestamp) -> np.ndarray:
    """
    Dokonuje predykcji zużycia energii dla podanych dat.

    :param poly: Obiekt przekształcenia cech.
    :param model: Wytrenowany model regresji.
    :param future_dates: Seria dat.
    :param min_date: Początkowa data użyta podczas trenowania modelu.
    :return: Predykcje zużycia energii.
    """
    logger.info("Dokonywanie predykcji na danych future_dates")
    future_days = (future_dates - min_date).days         # modyfikacja: używamy min_date z trenowania
    future_days = pd.DataFrame(future_days, columns=["Days"])  # zamiana na DataFrame z nazwą kolumny
    future_days_poly = poly.transform(future_days)
    predictions = model.predict(future_days_poly)
    return predictions
