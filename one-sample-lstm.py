import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ============================
# Set seeds & determinism
# ============================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# Data Loading and Preprocessing
# ============================
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)
    # drop label column if present
    return train_data.iloc[:, :-1], test_data.iloc[:, :-1]

def preprocess_data(data, scaler=None):
    arr = data.to_numpy()
    if scaler is None:
        scaler = StandardScaler().fit(arr)
    scaled = scaler.transform(arr)
    return scaled, scaler

# ============================
# Sequence Creation (seq_len=1 here)
# ============================
SEQ_LENGTH = 1

def create_sequences(data, seq_length=SEQ_LENGTH):
    # reshape (n_samples, n_features) → (n_samples, seq_length, n_features)
    return data.reshape((data.shape[0], seq_length, data.shape[1]))

# ============================
# LSTM Autoencoder Definition
# ============================
def build_lstm_autoencoder(train_sequences,
                           latent_dim=64,
                           epochs=50,
                           batch_size=32):
    """
    train_sequences: np.array of shape (n_samples, seq_len, n_features)
    """
    n_timesteps = train_sequences.shape[1]
    n_features  = train_sequences.shape[2]

    inputs = Input(shape=(n_timesteps, n_features))
    # Encoder
    encoded = LSTM(latent_dim, activation='relu')(inputs)
    # Decoder
    decoded = RepeatVector(n_timesteps)(encoded)
    decoded = LSTM(n_features, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        train_sequences, train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[es],
        verbose=2
    )
    return model

# ============================
# Anomaly Scoring & Evaluation
# ============================
def compute_reconstruction_error(model, sequences):
    """
    Returns a 1D array of MSE reconstruction error per sequence.
    """
    reconstructed = model.predict(sequences, verbose=0)
    # average over features & timesteps
    mse = np.mean(np.mean(np.square(sequences - reconstructed), axis=2), axis=1)
    return mse

def evaluate_balanced_lstm(model, normal_scaled, anomaly_scaled,
                           seq_length=SEQ_LENGTH,
                           threshold_percentile=80):
    start = time.time()

    # build sequences
    normal_seq  = create_sequences(normal_scaled, seq_length)
    anomaly_seq = create_sequences(anomaly_scaled, seq_length)

    # compute errors
    normal_err  = compute_reconstruction_error(model, normal_seq)
    anomaly_err = compute_reconstruction_error(model, anomaly_seq)

    # threshold on normal behavior
    threshold = np.percentile(normal_err, threshold_percentile)

    # classify
    normal_preds  = (normal_err  > threshold).astype(int)
    anomaly_preds = (anomaly_err > threshold).astype(int)

    # ground truths
    y_true = np.concatenate([
        np.zeros_like(normal_preds),
        np.ones_like(anomaly_preds)
    ])
    y_pred = np.concatenate([normal_preds, anomaly_preds])

    # metrics
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    inference_time = time.time() - start

    return f1, precision, recall, tn, fp, fn, tp, threshold, inference_time

# ============================
# Main Pipeline with LSTM
# ============================
def main_lstm(parent_dir, output_csv):
    # collect all dataset paths
    datasets = [
        os.path.join(parent_dir, d, f"{d}.csv")
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    results = []
    trained = {}

    # 1) Train one autoencoder per dataset
    for train_path in datasets:
        df = pd.read_csv(train_path).iloc[:, :-1]
        scaled, scaler = preprocess_data(df)
        seqs = create_sequences(scaled, SEQ_LENGTH)
        model = build_lstm_autoencoder(seqs)
        trained[train_path] = (model, scaler)

    # 2) Evaluate each pair (train vs anomaly)
    for train_path, anomaly_path in sorted(((a,b) for a in datasets for b in datasets)):
        if "rpi" in train_path or "rpi" in anomaly_path:
            continue

        model, scaler = trained[train_path]

        # single normal sample
        df_train = pd.read_csv(train_path).iloc[:, :-1]
        scaled_train, _ = preprocess_data(df_train, scaler)
        idx = np.random.choice(len(scaled_train), size=1, replace=False)
        normal_eval = scaled_train[idx]

        # 1 anomaly samples
        df_anom = pd.read_csv(anomaly_path).sample(n=1, random_state=SEED).iloc[:, :-1]
        anomaly_eval, _ = preprocess_data(df_anom, scaler)

        f1, prec, rec, tn, fp, fn, tp, thr, t_inf = evaluate_balanced_lstm(
            model, normal_eval, anomaly_eval
        )

        results.append({
            "Train Dataset":    os.path.basename(train_path),
            "Anomaly Dataset":  os.path.basename(anomaly_path),
            "tn":               tn,
            "tp":               tp,
            "fn":               fn,
            "fp":               fp,
        })

    print(np.average(t_inf))
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"LSTM‐based evaluation results saved to {output_csv}")


# Example usage:
if __name__ == "__main__":
    parent_dir = 'custom_datasets'
    output_csv = '1sample-lstm_evaluation_results.csv'
    main_lstm(parent_dir, output_csv)
