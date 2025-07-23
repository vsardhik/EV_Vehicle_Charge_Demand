import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load the preprocessed data
DATA_PATH = 'Project/preprocessed_ev_data.csv'
df = pd.read_csv(DATA_PATH)

# 2. Define features and target
features = [
    'months_since_start',
    'county_encoded',
    'ev_total_lag1',
    'ev_total_lag2',
    'ev_total_lag3',
    'ev_total_roll_mean_3',
    'ev_total_pct_change_1',
    'ev_total_pct_change_3',
    'ev_growth_slope',
]
target = 'Electric Vehicle (EV) Total'
X = df[features]
y = df[target]

# 3. Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

# 4. Model selection and hyperparameter tuning
param_dist = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', None]
}
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)
model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# 5. Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 6. Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title('EV Adoption Forecast: Actual vs Predicted')
plt.xlabel('Test Sample Index')
plt.ylabel('EV Total')
plt.legend()
plt.tight_layout()
plt.show()

# 7. Save the model
joblib.dump(model, 'ev_adoption_rf_model.joblib')
print('Model saved as ev_adoption_rf_model.joblib')

# 8. Forecasting future EV adoption (example: next 12 months for a sample county)
def forecast_next_months(df, model, county_name, months=12):
    """
    Forecast EV adoption for the next 'months' for a given county using the trained model.
    """
    # Get the latest row for the county
    county_df = df[df['County'] == county_name].sort_values('months_since_start')
    if county_df.empty:
        print(f"County '{county_name}' not found in data.")
        return
    last_row = county_df.iloc[-1].copy()
    forecasts = []
    for i in range(months):
        # Prepare input features for the next month
        input_features = [
            last_row['months_since_start'] + 1,
            last_row['county_encoded'],
            last_row['Electric Vehicle (EV) Total'],
            last_row['ev_total_lag1'],
            last_row['ev_total_lag2'],
            np.mean([
                last_row['Electric Vehicle (EV) Total'],
                last_row['ev_total_lag1'],
                last_row['ev_total_lag2']
            ]),
            (last_row['Electric Vehicle (EV) Total'] - last_row['ev_total_lag1']) / last_row['ev_total_lag1'] if last_row['ev_total_lag1'] else 0,
            (last_row['Electric Vehicle (EV) Total'] - last_row['ev_total_lag3']) / last_row['ev_total_lag3'] if last_row['ev_total_lag3'] else 0,
            last_row['ev_growth_slope'],
        ]
        pred = model.predict([input_features])[0]
        forecasts.append(pred)
        # Update lags for next iteration
        last_row['months_since_start'] += 1
        last_row['ev_total_lag3'] = last_row['ev_total_lag2']
        last_row['ev_total_lag2'] = last_row['ev_total_lag1']
        last_row['ev_total_lag1'] = last_row['Electric Vehicle (EV) Total']
        last_row['Electric Vehicle (EV) Total'] = pred
        last_row['ev_total_roll_mean_3'] = np.mean([
            last_row['Electric Vehicle (EV) Total'],
            last_row['ev_total_lag1'],
            last_row['ev_total_lag2']
        ])
        last_row['ev_total_pct_change_1'] = (last_row['Electric Vehicle (EV) Total'] - last_row['ev_total_lag1']) / last_row['ev_total_lag1'] if last_row['ev_total_lag1'] else 0
        last_row['ev_total_pct_change_3'] = (last_row['Electric Vehicle (EV) Total'] - last_row['ev_total_lag3']) / last_row['ev_total_lag3'] if last_row['ev_total_lag3'] else 0
        # Growth slope remains the same for simplicity
    return forecasts

# Example usage:
county_to_forecast = df['County'].iloc[0]  # Use the first county in the dataset
future_months = 12
future_forecast = forecast_next_months(df, model, county_to_forecast, months=future_months)
print(f"Forecast for next {future_months} months in county '{county_to_forecast}':")
print(future_forecast) 