import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import time
from sklearn.preprocessing import StandardScaler

PARENT_DIR = 'custom_datasets'
RESULTS_CSV = 'lstm_ae_anomaly_eval_single_window.csv'
SEQ_LEN = 3
STEP = 3
THRESHOLD_PCT = 95
SEED = 42
EPOCHS = 20
BATCH_SIZE = 32

np.random.seed(SEED)
tf.random.set_seed(SEED)


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
    return a 3D array of shape (N_windows, seq_len, F), where N_windows = floor((T − seq_len)/step) + 1.
    """
    T, F = arr.shape
    windows = []
    for start in range(0, T - seq_len + 1, step):
        block = arr[start: start + seq_len]   # shape = (seq_len, F)
        windows.append(block)
    if not windows:
        return np.empty((0, seq_len, F))
    return np.stack(windows)  # shape = (N_windows, seq_len, F)


def build_lstm_autoencoder(seq_len, feature_dim):
    """
    Build an LSTM‐based autoencoder for sequences of length seq_len and feature_dim features.
    """
    inp = Input(shape=(seq_len, feature_dim))
    encoded, state_h, state_c = LSTM(feature_dim * 2, return_state=True)(inp)
    latent = Dense(feature_dim, activation='relu')(state_h)
    repeated = RepeatVector(seq_len)(latent)
    decoded = LSTM(feature_dim * 2, return_sequences=True)(repeated)
    out = TimeDistributed(Dense(feature_dim))(decoded)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm_autoencoder(model, data):
    """
    Train the LSTM autoencoder on `data`, which is a numpy array of shape (N, seq_len, F).
    """
    model.fit(
        data,
        data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0
    )


def compute_reconstruction_errors(model, data):
    """
    Given a trained model and `data` (numpy array shape (N, seq_len, F)),
    return a 1D array of MSE per sequence.
    """
    recon = model.predict(data, verbose=0)
    mse = np.mean(np.square(data - recon), axis=(1, 2))
    return mse  # shape = (N,)


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
        train_windows = make_windows(train_arr, SEQ_LEN, STEP)  # (N_train, SEQ_LEN, F)
        if train_windows.size == 0:
            continue

        N_train, _, F = train_windows.shape
        train_flat = train_windows.reshape(N_train, SEQ_LEN * F)
        scaler = StandardScaler().fit(train_flat)
        train_scaled_flat = scaler.transform(train_flat)
        train_scaled = train_scaled_flat.reshape(N_train, SEQ_LEN, F)

        model = build_lstm_autoencoder(SEQ_LEN, F)
        train_lstm_autoencoder(model, train_scaled)

        mse_train = compute_reconstruction_errors(model, train_scaled)
        threshold = np.percentile(mse_train, THRESHOLD_PCT)

        for test_path in dataset_paths:
            test_arr = load_csv_drop_target(test_path)
            test_windows = make_windows(test_arr, SEQ_LEN, STEP)  # (N_test, SEQ_LEN, F)
            if test_windows.size == 0:
                print(f"[Skipped] {os.path.basename(test_path)} has fewer than {SEQ_LEN} rows.")
                continue

            rand_idx = np.random.randint(0, test_windows.shape[0])
            single_window = test_windows[rand_idx : rand_idx + 1]  # (1, SEQ_LEN, F)
            single_flat = single_window.reshape(1, SEQ_LEN * F)
            single_scaled_flat = scaler.transform(single_flat)
            single_scaled = single_scaled_flat.reshape(1, SEQ_LEN, F)

            start = time.time()
            mse_single = compute_reconstruction_errors(model, single_scaled)[0]
            inference_time = time.time() - start
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
    print(f"\nAll results (sorted) saved to '{RESULTS_CSV}'.")
