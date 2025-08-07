import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math

def transform_to_supervised(df, datetime_col, feature_cols, target_col, n_past=24, n_future=7):

    df = df.copy()
    df.sort_values(by=datetime_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    supervised = pd.DataFrame()

    for col in feature_cols:
        for i in range(n_past, 0, -1):
            supervised[f'{col}(t-{i})'] = df[col].shift(i)

    for i in range(0, n_future):
        supervised[f'{target_col}(t+{i})'] = df[target_col].shift(-i)

    supervised.dropna(inplace=True)

    return supervised

def compute_diff(df: pd.DataFrame) -> pd.DataFrame:
    
    return df.diff().dropna().reset_index(drop=True)


df = pd.read_csv('./data/clean/df_clean.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.reset_index(drop=True).sort_values(by='DATE')
df_snowlvl = df[['DATE', 'PRENEI_Q', 'PRELIQ_Q', 'T_Q', 'FF_Q', 'Q_Q',
       'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q', 'ETP_Q', 'PE_Q', 'SWI_Q',
       'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE_Q', 'RESR_NEIGE6_Q', 'HTEURNEIGE_Q',
       'HTEURNEIGE6_Q', 'HTEURNEIGEX_Q', 'SNOW_FRAC_Q', 'ECOULEMENT_Q',
       'WG_RACINE_Q', 'WGI_RACINE_Q', 'TINF_H_Q', 'TSUP_H_Q']]

df_supervised = transform_to_supervised(df_snowlvl, 'DATE', ['HTEURNEIGE_Q'], 'HTEURNEIGE_Q', n_past=24, n_future=24)

df_diff = compute_diff(df_supervised.copy())
X = df_diff.iloc[:,:-24].copy()
y = df_diff.iloc[:,-24:].copy()
X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
y_train, y_test = y[:int(len(X)*0.8)], y[int(len(X)*0.8):]

def train_forecast_per_step(X_train, X_test, y_train, y_test, df_supervised, split_idx):
    results = {}
    scaler_x = StandardScaler()

    # Scale inputs
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    for step in range(y_train.shape[1]):
        # Fit a separate scaler for each output step
        scaler_yi = StandardScaler()
        y_train_i_scaled = scaler_yi.fit_transform(y_train.iloc[:, step].values.reshape(-1, 1))
        y_test_i_scaled = scaler_yi.transform(y_test.iloc[:, step].values.reshape(-1, 1))

        # Train model for this step
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
        model.fit(X_train_scaled, y_train_i_scaled.ravel())

        # Predict
        yhat = model.predict(X_test_scaled)
        ytrue = y_test_i_scaled.ravel()

        # Evaluate
        rmse = math.sqrt(mean_squared_error(ytrue, yhat))
        print(f"Step {step} - RMSE (diff): {rmse:.4f}")

        # Plot predicted vs true (diff values)
        # plt.figure(figsize=(15, 3))
        # plt.plot(ytrue, label='True Δ')
        # plt.plot(yhat, label='Predicted Δ', alpha=0.7)
        # plt.title(f"Step {step} - Diff values")
        # plt.legend()
        # plt.show()

        # Reconstruct absolute values
        last_values = df_supervised.iloc[split_idx + 1:, -25].values.reshape(-1, 1)  # value(t-1)
        yhat_inv = scaler_yi.inverse_transform(yhat.reshape(-1, 1))
        ytrue_inv = scaler_yi.inverse_transform(ytrue.reshape(-1, 1))
        yhat_abs = last_values + yhat_inv
        ytrue_abs = last_values + ytrue_inv

        # Plot reconstructed absolute values
        # plt.figure(figsize=(15, 3))
        # plt.plot(ytrue_abs, label='True')
        # plt.plot(yhat_abs, label='Predicted', alpha=0.7)
        # plt.title(f"Step {step} - Reconstructed absolute values")
        # plt.legend()
        # plt.show()

        # Save results
        results[f"step_{step}"] = {
            "model": model,
            "scaler_y": scaler_yi,
            "rmse": rmse,
            "yhat_diff": yhat,
            "ytrue_diff": ytrue,
            "yhat_abs": yhat_abs,
            "ytrue_abs": ytrue_abs,
        }

    return results

results = train_forecast_per_step(X_train,X_test,y_train,y_test, df_supervised, split_idx=int(len(X)*0.8))

prediction = []
for step in results.keys():
    prediction.append(results[step]['yhat_abs'])

idx = 70
plt.plot(range(24), df_supervised.iloc[int(len(X)*0.8) + idx,:-24])
plt.plot(range(24,48), df_supervised.iloc[int(len(X)*0.8) + idx,-24:])
plt.plot(range(24,48), [pred[idx] for pred in prediction]);