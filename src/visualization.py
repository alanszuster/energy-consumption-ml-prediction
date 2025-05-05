import matplotlib.pyplot as plt
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_consumption(df: pd.DataFrame) -> None:
    """
    Rysuje zużycie energii w czasie.

    :param df: DataFrame zawierający dane.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Consumption'], marker='o', color='b', label='Zużycie energii')
    plt.title('Zużycie energii w czasie')
    plt.xlabel('Data')
    plt.ylabel('Zużycie energii (kWh)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_costs(df: pd.DataFrame) -> None:
    """
    Rysuje koszty energii w czasie.

    :param df: DataFrame zawierający dane.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Cost'], marker='o', color='r', label='Koszty energii')
    plt.title('Koszty energii w czasie')
    plt.xlabel('Data')
    plt.ylabel('Koszt (zł)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_predictions(future_dates, predictions) -> None:
    """
    Rysuje prognozowane zużycie energii.

    :param future_dates: Daty przyszłe.
    :param predictions: Prognozy zużycia.
    """
    plt.plot(future_dates, predictions, marker='o', color='g', label='Prognozowane zużycie')
    plt.title('Prognoza zużycia energii na kolejne miesiące')
    plt.xlabel('Data')
    plt.ylabel('Zużycie energii (kWh)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

def display_monthly_summary(dates, predictions) -> None:
    """
    Wyświetla prognozę zużycia energii i kosztów na kolejne 12 miesięcy.

    :param dates: Lista dat.
    :param predictions: Prognozy zużycia energii.
    """
    COST_PER_KWH = 0.75
    logger.info("Wyświetlanie miesięcznego podsumowania prognoz")
    print("\nPrognoza zużycia energii i kosztów na kolejne 12 miesięcy:")
    print("---------------------------------------------------------")
    for date, consumption in zip(dates, predictions):
        cost = consumption * COST_PER_KWH
        print(f"{date.strftime('%B %Y')} - Zużycie: {consumption:.2f} kWh / Koszt: {cost:.2f} zł")
