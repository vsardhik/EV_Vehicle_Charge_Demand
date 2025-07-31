import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# --- Configuration ---
DATA_PATH = os.getenv('EV_DATA_PATH', 'preprocessed_ev_data.csv')
MODEL_PATH = os.getenv('EV_MODEL_PATH', 'ev_adoption_rf_model.joblib')
TEST_SIZE = 0.1
RANDOM_STATE = 42
FUTURE_MONTHS = 12

def load_data(path):
    """Load preprocessed EV data from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)

def get_features_and_target(df):
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
    return df[features], df[target]

def train_test_split_time_series(X, y, test_size=0.1):
    """Split data for time series (no shuffle)."""
    return train_test_split(X, y, shuffle=False, test_size=test_size)

def tune_model(X_train, y_train, random_state=42):
    """Randomized hyperparameter search for RandomForestRegressor."""
    param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 4, 6, 8],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2', None]
    }
    rf = RandomForestRegressor(random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=random_state
    )
    random_search.fit(X_train, y_train)
    print("Best Parameters:", random_search.best_params_)
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    return y_pred

def plot_actual_vs_predicted(y_test, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title('EV Adoption Forecast: Actual vs Predicted')
    plt.xlabel('Test Sample Index')
    plt.ylabel('EV Total')
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_model(model, path):
    """Save the trained model to disk."""
    joblib.dump(model, path)
    print(f'Model saved as {path}')

def forecast_next_months(df, model, county_name, months=12):
    """
    Forecast EV adoption for the next 'months' for a given county using the trained model.
    """
    county_df = df[df['County'] == county_name].sort_values('months_since_start')
    if county_df.empty:
        print(f"County '{county_name}' not found in data.")
        return []
    last_row = county_df.iloc[-1].copy()
    forecasts = []
    for _ in range(months):
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

def main():
    # Load data
    df = load_data(DATA_PATH)
    X, y = get_features_and_target(df)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=TEST_SIZE)
    # Model selection and training
    model = tune_model(X_train, y_train, random_state=RANDOM_STATE)
    # Evaluation
    y_pred = evaluate_model(model, X_test, y_test)
    plot_actual_vs_predicted(y_test, y_pred)
    # Save model
    save_model(model, MODEL_PATH)
    # Forecast future
    county_to_forecast = df['County'].iloc[0]  # Use the first county in the dataset
    future_forecast = forecast_next_months(df, model, county_to_forecast, months=FUTURE_MONTHS)
    print(f"Forecast for next {FUTURE_MONTHS} months in county '{county_to_forecast}':")
    print(future_forecast)

if __name__ == "__main__":
    main()
