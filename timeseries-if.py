import os
import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest   # added IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

PARENT_DIR = 'custom_datasets'
RESULTS_CSV = 'rf_anomaly_eval_single_window.csv'
SEQ_LEN = 3
STEP = 3
SEED = 42

np.random.seed(SEED)


def load_csv_drop_target(path):
    """
    Load a CSV into a pandas DataFrame, drop any 'label' or 'Target' column if present,
    and return the remaining values as a numpy array.
    """
    df = pd.read_csv(path)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])
    return df.to_numpy()


def make_windows(arr, seq_len=SEQ_LEN, step=STEP):
    """
    Given a (T × F) array, extract non-overlapping windows of length seq_len (STEP = seq_len),
    flatten each window into a (seq_len * F) vector, and return a 2D array of shape
      (N_windows, seq_len * F).
    """
    T, F = arr.shape
    windows = []
    for start in range(0, T - seq_len + 1, step):
        block = arr[start: start + seq_len]
        windows.append(block.flatten())
    return np.vstack(windows)


def train_random_forest(X_train, y_train):
    """
    Fit a RandomForestClassifier on (X_train, y_train).
    Returns the trained model.
    """
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


if __name__ == "__main__":
    dataset_paths = []
    for folder in sorted(os.listdir(PARENT_DIR)):
        folder_dir = os.path.join(PARENT_DIR, folder)
        csv_path = os.path.join(folder_dir, f"{folder}.csv")
        if os.path.isdir(folder_dir) and os.path.isfile(csv_path):
            dataset_paths.append(csv_path)

    if not dataset_paths:
        raise FileNotFoundError(
            f"No CSV files found in {PARENT_DIR!r} with the expected structure.")

    results = []

    for train_path in dataset_paths:
        train_arr = load_csv_drop_target(train_path)
        train_windows = make_windows(train_arr, SEQ_LEN, STEP)
        # **All train windows are normal**
        scaler = StandardScaler().fit(train_windows)
        train_scaled = scaler.transform(train_windows)

        # train an IsolationForest **only** on the normal windows
        iso_model = IsolationForest(random_state=SEED, contamination='auto')
        iso_model.fit(train_scaled)

        for test_path in dataset_paths:
            test_arr = load_csv_drop_target(test_path)
            test_windows = make_windows(test_arr, SEQ_LEN, STEP)

            if test_windows.shape[0] == 0:
                print(
                    f"  [Skipped] {os.path.basename(test_path)} has fewer than {SEQ_LEN} rows (no windows).")
                continue

            # pick one random window to evaluate
            rand_idx = np.random.randint(low=0, high=test_windows.shape[0])
            single_window = test_windows[rand_idx: rand_idx + 1]
            single_scaled = scaler.transform(single_window)

            # inference with IsolationForest
            start = time.time()
            pred = iso_model.predict(single_scaled)[0]
            inference_time = time.time() - start

            # IsolationForest: -1=anomaly, +1=normal
            if pred == -1:
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
    results_df.sort_values(by=['Train CSV', 'Test CSV'],
                           inplace=True, ignore_index=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll results (sorted) saved to {RESULTS_CSV!r}.")
