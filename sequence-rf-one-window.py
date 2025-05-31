import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

PARENT_DIR = 'custom_datasets'
RESULTS_CSV = 'rf_anomaly_eval_single_window.csv'
SEQ_LEN = 30
STEP = 30
SEED = 42

np.random.seed(SEED)


def load_csv_drop_target(path):
    """
    Load a CSV into a pandas DataFrame, drop any 'label' or 'Target' column if present,
    and return the remaining values as a numpy array.

    We assume each row is one timestamp, each remaining column is a genuine feature.
    """
    df = pd.read_csv(path)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])
    return df.to_numpy()  # shape = (T, F)


def make_windows(arr, seq_len=SEQ_LEN, step=STEP):
    """
    Given a (T × F) array, extract non-overlapping windows of length seq_len (STEP = seq_len),
    flatten each window into a (seq_len * F) vector, and return a 2D array of shape
      (N_windows, seq_len * F),
    where N_windows = floor((T − seq_len)/step) + 1.
    """
    T, F = arr.shape
    windows = []
    for start in range(0, T - seq_len + 1, step):
        block = arr[start: start + seq_len]   # shape = (seq_len, F)
        windows.append(block.flatten())        # shape = (seq_len * F,)
    # shape = (N_windows, seq_len * F)
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
        train_arr = load_csv_drop_target(
            train_path)             # shape = (T_train, F)
        # shape = (N_train_windows, SEQ_LEN*F)
        train_windows = make_windows(train_arr, SEQ_LEN, STEP)

        # label all train windows as class 0
        y_train_windows = np.zeros(train_windows.shape[0], dtype=int)

        for test_path in dataset_paths:
            test_arr = load_csv_drop_target(
                test_path)              # shape = (T_test, F)
            # shape = (N_test_windows, SEQ_LEN*F)
            test_windows = make_windows(test_arr, SEQ_LEN, STEP)

            if test_windows.shape[0] == 0:
                print(
                    f"  [Skipped] {os.path.basename(test_path)} has fewer than {SEQ_LEN} rows (no windows).")
                continue

            rand_idx = np.random.randint(low=0, high=test_windows.shape[0])
            # shape = (1, SEQ_LEN*F)
            single_window = test_windows[rand_idx: rand_idx + 1]

            # build supervised training set: train_windows=0, other_test_windows=1
            mask = np.ones(test_windows.shape[0], dtype=bool)
            mask[rand_idx] = False
            # shape = (N_test_windows - 1, SEQ_LEN*F)
            other_test_windows = test_windows[mask]
            y_other_test = np.ones(other_test_windows.shape[0], dtype=int)

            X_train = np.vstack([train_windows, other_test_windows])
            y_train = np.concatenate([y_train_windows, y_other_test])

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            single_window_scaled = scaler.transform(single_window)

            rf_model = train_random_forest(X_train_scaled, y_train)

            test_pred = int(rf_model.predict(single_window_scaled)[0])
            if test_pred == 1:
                TP = 1
                FN = 0
            else:
                TP = 0
                FN = 1

            start_row = rand_idx * STEP

            y_true_comb = np.concatenate([
                np.zeros(train_windows.shape[0], dtype=int),
                np.ones(1, dtype=int)
            ])

            results.append({
                'Train CSV': os.path.basename(train_path),
                'Test CSV': os.path.basename(test_path),
                'TN': "0",
                'TP': TP,
                'FN': FN,
                'FP': "0",
            })

    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['Train CSV', 'Test CSV'],
                           inplace=True, ignore_index=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll results (sorted) saved to {RESULTS_CSV!r}.")
