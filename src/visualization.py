import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8')

def plot_consumption(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(df['Date'], df['Consumption'], marker='o', linewidth=2, color='blue')
    ax1.set_title('Energy Consumption Over Time')
    ax1.set_ylabel('Consumption (kWh)')
    ax1.grid(True, alpha=0.3)

    monthly_avg = df.groupby(df['Date'].dt.to_period('M'))['Consumption'].mean()
    monthly_dates = [period.start_time for period in monthly_avg.index]

    ax2.bar(monthly_dates, monthly_avg.values, alpha=0.7, color='green', width=20)
    ax2.set_title('Monthly Average Consumption')
    ax2.set_ylabel('Average (kWh)')
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('energy_consumption_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_costs(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['Date'], df['Cost'], marker='s', linewidth=2, color='red')
    ax.set_title('Energy Costs Over Time')
    ax.set_ylabel('Cost (PLN)')
    ax.grid(True, alpha=0.3)

    total_cost = df['Cost'].sum()
    avg_cost = df['Cost'].mean()

    ax.text(0.02, 0.98, f'Total: {total_cost:.0f} PLN\nAverage: {avg_cost:.0f} PLN',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('energy_costs_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(predictions_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(predictions_df['Date'], predictions_df['Predicted_Consumption'],
             marker='o', linewidth=2, color='orange')
    ax1.set_title('Energy Consumption Predictions - Next 12 Months')
    ax1.set_ylabel('Predicted Consumption (kWh)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(predictions_df['Date'], predictions_df['Predicted_Cost'],
             marker='s', linewidth=2, color='purple')
    ax2.set_title('Energy Cost Predictions - Next 12 Months')
    ax2.set_ylabel('Predicted Cost (PLN)')
    ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('energy_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def display_monthly_summary(predictions_df):
    print("\n" + "="*60)
    print("ENERGY CONSUMPTION PREDICTIONS - NEXT 12 MONTHS")
    print("="*60)

    total_consumption = predictions_df['Predicted_Consumption'].sum()
    total_cost = predictions_df['Predicted_Cost'].sum()
    avg_consumption = predictions_df['Predicted_Consumption'].mean()
    avg_cost = predictions_df['Predicted_Cost'].mean()

    print(f"\nSUMMARY:")
    print(f"Total Predicted Consumption: {total_consumption:.0f} kWh")
    print(f"Total Predicted Cost: {total_cost:.0f} PLN")
    print(f"Average Monthly Consumption: {avg_consumption:.0f} kWh")
    print(f"Average Monthly Cost: {avg_cost:.0f} PLN")

    print(f"\nMONTHLY BREAKDOWN:")
    print("-" * 60)
    print(f"{'Month':<15} {'Consumption':<12} {'Cost':<10} {'Season'}")
    print("-" * 60)

    seasons = {12: 'Winter', 1: 'Winter', 2: 'Winter',
               3: 'Spring', 4: 'Spring', 5: 'Spring',
               6: 'Summer', 7: 'Summer', 8: 'Summer',
               9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

    for _, row in predictions_df.iterrows():
        month = row['Date'].strftime('%B %Y')
        consumption = row['Predicted_Consumption']
        cost = row['Predicted_Cost']
        season = seasons[row['Date'].month]
        print(f"{month:<15} {consumption:>8.0f} kWh {cost:>8.0f} PLN {season:>8}")

    print("-" * 60)

def plot_model_performance(predictor, historical_data, performance):
    fig = plt.figure(figsize=(15, 10))

    # Model metrics
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['R² Score', 'RMSE', 'MAE']
    values = [performance['r2_score'], performance['rmse'], performance['mae']]

    bars = ax1.bar(metrics, values, color=['blue', 'red', 'green'], alpha=0.7)
    ax1.set_title('Model Performance')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom')

    # Feature importance
    ax2 = plt.subplot(2, 3, 2)
    feature_importance = predictor.get_feature_importance()
    if feature_importance:
        features = list(feature_importance.keys())[:6]
        importance_values = list(feature_importance.values())[:6]
        ax2.barh(features, importance_values, color='orange', alpha=0.7)
        ax2.set_title('Top Features')
    else:
        ax2.text(0.5, 0.5, 'Not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Feature Importance')

    # Consumption distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(historical_data['Consumption'], bins=10, alpha=0.7, color='cyan', edgecolor='black')
    ax3.set_title('Consumption Distribution')
    ax3.set_xlabel('Consumption (kWh)')

    # Seasonal pattern
    ax4 = plt.subplot(2, 3, 4)
    monthly_pattern = historical_data.groupby('Month')['Consumption'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ax4.plot(range(1, 13), monthly_pattern, marker='o', linewidth=2)
    ax4.set_title('Seasonal Pattern')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(months, rotation=45)

    # Model comparison (if available)
    ax5 = plt.subplot(2, 3, 5)
    if 'all_models' in performance:
        model_names = list(performance['all_models'].keys())
        model_scores = [performance['all_models'][name]['r2_score'] for name in model_names]
        ax5.bar(model_names, model_scores, color=['blue', 'red', 'green'], alpha=0.7)
        ax5.set_title('Model Comparison')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax5.text(0.5, 0.5, f'Best: {performance["model_name"]}',
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Best Model')

    # Historical trend
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(historical_data['Date'], historical_data['Consumption'], linewidth=2, alpha=0.7)
    ax6.set_title('Historical Trend')
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
