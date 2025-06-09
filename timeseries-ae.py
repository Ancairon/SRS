import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

PARENT_DIR = 'custom_datasets'
RESULTS_CSV = 'ae_anomaly_eval_single_window.csv'
SEQ_LEN = 10
STEP = 10
THRESHOLD_PCT = 95
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_csv_drop_target(path):
    df = pd.read_csv(path)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])
    return df.to_numpy()


def make_windows(arr, seq_len=SEQ_LEN, step=STEP):
    T, F = arr.shape
    windows = []
    for start in range(0, T - seq_len + 1, step):
        block = arr[start: start + seq_len]
        windows.append(block.flatten())
    return np.vstack(windows)


def build_autoencoder(input_dim):
    inp = Input(shape=(input_dim,))
    encoded = Dense(input_dim // 2, activation='relu')(inp)
    encoded = Dense(input_dim // 4, activation='relu')(encoded)
    decoded = Dense(input_dim // 2, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    ae = Model(inputs=inp, outputs=decoded)
    ae.compile(optimizer='adam', loss='mse')
    return ae


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
        train_arr = load_csv_drop_target(train_path)
        train_windows = make_windows(train_arr, SEQ_LEN, STEP)
        scaler = StandardScaler().fit(train_windows)
        train_windows_scaled = scaler.transform(train_windows)

        input_dim = train_windows_scaled.shape[1]
        ae_model = build_autoencoder(input_dim)
        ae_model.fit(
            train_windows_scaled,
            train_windows_scaled,
            epochs=20,
            batch_size=32,
            shuffle=True,
            verbose=0
        )

        recon_train = ae_model.predict(train_windows_scaled, verbose=0)
        mse_train = np.mean(np.square(recon_train - train_windows_scaled), axis=1)
        threshold = np.percentile(mse_train, THRESHOLD_PCT)

        for test_path in dataset_paths:
            test_arr = load_csv_drop_target(test_path)
            test_windows = make_windows(test_arr, SEQ_LEN, STEP)

            if test_windows.shape[0] == 0:
                print(f"[Skipped] {os.path.basename(test_path)} has fewer than {SEQ_LEN} rows.")
                continue

            rand_idx = np.random.randint(low=0, high=test_windows.shape[0])
            single_window = test_windows[rand_idx: rand_idx + 1]
            single_scaled = scaler.transform(single_window)
            
            start = time.time()
            recon_single = ae_model.predict(single_scaled, verbose=0)
            inference_time = time.time() - start

            mse_single = np.mean(np.square(recon_single - single_scaled))
            pred = 1 if mse_single > threshold else 0
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
    print(f"\nAll results (sorted) saved to {RESULTS_CSV!r}.")
