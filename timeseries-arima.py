import os
import numpy as np
import time
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler

PARENT_DIR = 'custom_datasets'
RESULTS_CSV = 'arima_anomaly_eval_single_window.csv'
SEQ_LEN = 3
STEP = 3
THRESHOLD_PCT = 95
SEED = 42

np.random.seed(SEED)


def load_csv_drop_target(path):
    df = pd.read_csv(path)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])
    return df.to_numpy()  # shape = (T, F)


def make_windows(arr, seq_len=SEQ_LEN, step=STEP):
    T, F = arr.shape
    windows = []
    for start in range(0, T - seq_len + 1, step):
        block = arr[start : start + seq_len]  # shape = (seq_len, F)
        windows.append(block)
    if not windows:
        return np.empty((0, seq_len, F))
    return np.stack(windows)  # shape = (N_windows, seq_len, F)


if __name__ == "__main__":
    dataset_paths = []
    for folder in sorted(os.listdir(PARENT_DIR)):
        folder_dir = os.path.join(PARENT_DIR, folder)
        csv_path = os.path.join(folder_dir, f"{folder}.csv")
        if os.path.isdir(folder_dir) and os.path.isfile(csv_path):
            dataset_paths.append(csv_path)

    if not dataset_paths:
        raise FileNotFoundError(f"No CSV files found in {PARENT_DIR!r}.")

    results = []

    for train_path in dataset_paths:
        train_arr = load_csv_drop_target(train_path)  # shape = (T_train, F)
        if train_arr.shape[0] < SEQ_LEN:
            continue

        F = train_arr.shape[1]
        arima_models = []
        residual_sums = []

        # Fit ARIMA per feature, ensuring residuals match full length
        for c in range(F):
            series = train_arr[:, c]
            fitted = False
            for order in [(1, 0, 1), (1, 0, 0)]:
                try:
                    model = ARIMA(series, order=order).fit(disp=False)
                    resid = model.resid  # residuals of length T_train
                    if len(resid) == len(series):
                        arima_models.append(model)
                        residual_sums.append(resid**2)
                        fitted = True
                        break
                except Exception:
                    continue
            if not fitted:
                # If both orders fail, use zeros
                arima_models.append(None)
                residual_sums.append(np.zeros_like(series))

        residual_sums = np.stack(residual_sums, axis=1)  # shape = (T_train, F)
        sum_per_timestep = residual_sums.sum(axis=1)     # shape = (T_train,)
        threshold_per_step = np.percentile(sum_per_timestep, THRESHOLD_PCT)
        window_threshold = threshold_per_step * SEQ_LEN

        for test_path in dataset_paths:
            test_arr = load_csv_drop_target(test_path)  # shape = (T_test, F)
            windows = make_windows(test_arr, SEQ_LEN, STEP)
            if windows.size == 0:
                print(f"[Skipped] {os.path.basename(test_path)} has fewer than {SEQ_LEN} rows.")
                continue

            rand_idx = np.random.randint(0, windows.shape[0])
            single_window = windows[rand_idx]  # shape = (SEQ_LEN, F)

            # Forecast next SEQ_LEN for each feature from train series
            inference_time = 0
            start = time.time()

            forecasts = np.zeros((SEQ_LEN, F))
            for c in range(F):
                model = arima_models[c]
                if model is not None:
                    try:
                        fc = model.forecast(steps=SEQ_LEN)[0]
                    except Exception as e:
                        fc = np.zeros(SEQ_LEN)
                else:
                    fc = np.zeros(SEQ_LEN)
                forecasts[:, c] = fc
            inference_time = time.time() - start

            se = (single_window - forecasts) ** 2  # shape = (SEQ_LEN, F)
            window_error = se.sum()  # scalar

            pred = 1 if window_error > window_threshold else 0
            if pred == 1:
                TP = 1
                FN = 0
            else:
                TP = 0
                FN = 1

            results.append({
                'Train CSV': os.path.basename(train_path),
                'Test CSV': os.path.basename(test_path),
                'TN': "0",
                'TP': TP,
                'FN': FN,
                'FP': "0",
            })

    print(np.average(inference_time))
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['Train CSV', 'Test CSV'], inplace=True, ignore_index=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll results (sorted) saved to '{RESULTS_CSV}'.")
