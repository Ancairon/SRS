import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PARENT_DIR    = 'custom_datasets'
SUMMARY_CSV   = 'rf_anomaly_eval_summary.csv'
SEED          = 42

# Change this value to tweak the RF’s decision threshold:
#   - 0.5 means “predict class 1 if P(class=1) ≥ 0.5” (standard),
#   - setting to 0.7 → “predict 1 only if P(class=1) ≥ 0.7” (stricter, fewer positives),
#   - setting to 0.3 → “predict 1 if P(class=1) ≥ 0.3” (more lenient, more positives).
RF_THRESHOLD = 0.999

np.random.seed(SEED)


def load_csv_drop_target(path):
    """
    Load a CSV, drop any 'label' or 'Target' column if present,
    and return the remaining values as a numpy array of shape (T, F).
    """
    df = pd.read_csv(path)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])
    return df.to_numpy()  # shape = (T, F)


def make_windows(arr, seq_len, step):
    """
    Given a (T × F) array, extract non‐overlapping windows of length seq_len (STEP = step),
    flatten each window into a (seq_len * F) vector, and return a 2D array of shape
      (N_windows, seq_len * F),
    where N_windows = floor((T − seq_len)/step) + 1.
    """
    T, F = arr.shape
    windows = []
    for start in range(0, T - seq_len + 1, step):
        block = arr[start: start + seq_len]  # shape = (seq_len, F)
        windows.append(block.flatten())       # shape = (seq_len * F,)
    if not windows:
        return np.empty((0, seq_len * F))
    return np.vstack(windows)  # shape = (N_windows, seq_len * F)


def train_random_forest(X_train, y_train):
    """
    Fit a RandomForestClassifier on (X_train, y_train).
    Returns the trained model.
    """
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


def run(seq_len, step):
    """
    For a given window length (seq_len) and step, run the one-vs-all RF evaluation
    across all CSVs under PARENT_DIR, and return the aggregated confusion matrix:
      TN, FP, FN, TP
    """
    # Gather all CSV file paths under PARENT_DIR/{folder}/{folder}.csv
    dataset_paths = []
    for folder in sorted(os.listdir(PARENT_DIR)):
        folder_dir = os.path.join(PARENT_DIR, folder)
        csv_path  = os.path.join(folder_dir, f"{folder}.csv")
        if os.path.isdir(folder_dir) and os.path.isfile(csv_path):
            dataset_paths.append(csv_path)

    if not dataset_paths:
        raise FileNotFoundError(f"No CSV files found in {PARENT_DIR!r} with the expected structure.")

    # Counters for the confusion-matrix sums
    TN_sum = 0
    FP_sum = 0
    FN_sum = 0
    TP_sum = 0

    for train_path in dataset_paths:
        train_arr     = load_csv_drop_target(train_path)            # shape = (T_train, F)
        train_windows = make_windows(train_arr, seq_len, step)      # shape = (N_train, seq_len*F)

        if train_windows.shape[0] == 0:
            # Skip this train file if fewer than seq_len rows
            continue

        # All train-windows are labeled “0” (normal)
        y_train_windows = np.zeros(train_windows.shape[0], dtype=int)

        for test_path in dataset_paths:
            test_arr     = load_csv_drop_target(test_path)           # shape = (T_test, F)
            test_windows = make_windows(test_arr, seq_len, step)     # shape = (N_test, seq_len*F)

            if test_windows.shape[0] == 0:
                # Skip this test file if fewer than seq_len rows
                continue

            # Pick one random window from the test set to classify
            rand_idx       = np.random.randint(low=0, high=test_windows.shape[0])
            single_window  = test_windows[rand_idx: rand_idx + 1]     # shape = (1, seq_len*F)

            # The remaining test-windows are “anomalies” (label=1)
            mask            = np.ones(test_windows.shape[0], dtype=bool)
            mask[rand_idx]  = False
            other_test_wins = test_windows[mask]                     # shape = (N_test-1, seq_len*F)
            y_other_test    = np.ones(other_test_wins.shape[0], dtype=int)

            # Combine (train-normal=0) + (other-test=1) to form supervised training set
            X_train = np.vstack([train_windows, other_test_wins])
            y_train = np.concatenate([y_train_windows, y_other_test])

            # Standardize
            scaler                = StandardScaler().fit(X_train)
            X_train_scaled        = scaler.transform(X_train)
            single_window_scaled  = scaler.transform(single_window)

            # Train RF
            rf_model = train_random_forest(X_train_scaled, y_train)

            # Instead of `predict()`, get probability of class=1 and apply RF_THRESHOLD:
            proba_pos = rf_model.predict_proba(single_window_scaled)[0, 1]
            test_pred = 1 if proba_pos >= RF_THRESHOLD else 0

            # The true label for that single window is always 1
            if test_pred == 1:
                TP_sum += 1
            elif test_pred == 1 and (train_path == test_path or ("idle" in train_path and "idle" in test_path)):
                # print("correcting to FP")
                FP_sum +=1
            elif test_pred == 0 and (train_path == test_path or ("idle" in train_path and "idle" in test_path)):
                # print("correcting to TN")
                TN_sum +=1
            else:
                FN_sum += 1

            # (We never predict negative “normal” windows here by design.
            #  Hence TN_sum and FP_sum remain zero in this pipeline.)

    return TN_sum, FP_sum, FN_sum, TP_sum


if __name__ == "__main__":
    summary = []

    for i in range(2, 10, 1):
        print(f"Running SEQ_LEN={i}, STEP={i}  (Threshold={RF_THRESHOLD})")
        tn, fp, fn, tp = run(i, i)
        summary.append({
            'SEQ_LEN': i,
            'STEP': i,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary confusion‐matrix CSV saved to '{SUMMARY_CSV}'.")
