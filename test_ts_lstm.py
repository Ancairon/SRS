import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# =========================
# Configuration Parameters
# =========================
DATASETS_DIR = "custom_datasets"  # directory containing subfolders of datasets
WINDOW_SIZE = 10                  # number of time steps per window
VAL_SPLIT = 0.1                   # fraction of windows for validation
BATCH_SIZE = 32
EPOCHS = 50
LATENT_DIM = 64

MODEL_SAVE_DIR = "models"         # directory to save trained models
THRESHOLD_SAVE_DIR = "thresholds" # directory to save thresholds
RESULTS_CSV_PATH = "ATESTresults.csv"  # path to save inference results

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(THRESHOLD_SAVE_DIR, exist_ok=True)

# =========================
# Data Preparation
# =========================
def load_and_window(csv_path, window_size=WINDOW_SIZE):
    """
    Load a CSV file and slice into non-overlapping windows, dropping "Target" column if present.
    Returns: np.ndarray of shape (n_windows, window_size, n_features)
    """
    df = pd.read_csv(csv_path)
    if "Target" in df.columns:
        df = df.drop(columns=["Target"])
    data = df.values
    num_seconds, num_features = data.shape
    n_windows = num_seconds // window_size
    data = data[: n_windows * window_size, :]
    windows = data.reshape(n_windows, window_size, num_features)
    return windows

# =========================
# Model Definition
# =========================
def build_lstm_autoencoder(window_size, num_features, latent_dim=LATENT_DIM):
    """
    Build an LSTM autoencoder: (window_size, num_features) -> (window_size, num_features)
    """
    input_seq = layers.Input(shape=(window_size, num_features))
    encoded = layers.LSTM(latent_dim, activation='tanh')(input_seq)
    decoder_input = layers.RepeatVector(window_size)(encoded)
    decoded = layers.LSTM(num_features, activation='tanh', return_sequences=True)(decoder_input)
    autoencoder = models.Model(inputs=input_seq, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# =========================
# Threshold Computation
# =========================
def compute_threshold(model, X_val):
    """
    Compute reconstruction MSEs on X_val and return the 95th percentile as threshold.
    """
    X_pred = model.predict(X_val, verbose=0)
    mse = np.mean((X_val - X_pred) ** 2, axis=(1, 2))
    return float(np.percentile(mse, 95))

# =========================
# Inference
# =========================
def is_window_anomaly(window, model, threshold):
    """
    Given one window (window_size, n_features), return (is_anomaly, mse).
    """
    w = window.reshape(1, window.shape[0], window.shape[1])
    reconstructed = model.predict(w, verbose=0)
    mse = float(np.mean((w - reconstructed) ** 2))
    return (mse > threshold), mse

# =========================
# Main Pipeline
# =========================
def main():
    # Identify datasets
    dataset_names = [
        name for name in os.listdir(DATASETS_DIR)
        if os.path.isdir(os.path.join(DATASETS_DIR, name))
        and os.path.isfile(os.path.join(DATASETS_DIR, name, f"{name}.csv"))
    ]
    results = []

    for train_name in sorted(dataset_names):
        print("TRAIN FOR",train_name)
        train_csv = os.path.join(DATASETS_DIR, train_name, f"{train_name}.csv")
        windows = load_and_window(train_csv, WINDOW_SIZE)
        if windows.shape[0] < 2:
            continue  # skip if not enough windows

        # Split into train/val for threshold
        X_train, X_val = train_test_split(windows, test_size=VAL_SPLIT, random_state=42)

        # Build and train model
        num_features = windows.shape[2]
        model = build_lstm_autoencoder(WINDOW_SIZE, num_features)
        model.fit(
            X_train, X_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, X_val),
            verbose=0
        )

        # Compute and save threshold
        threshold = compute_threshold(model, X_val)
        model_path = os.path.join(MODEL_SAVE_DIR, f"ae_{train_name}.h5")
        thresh_path = os.path.join(THRESHOLD_SAVE_DIR, f"thresh_{train_name}.npy")
        model.save(model_path, include_optimizer=False)
        np.save(thresh_path, np.array([threshold]))

        for test_name in sorted(dataset_names):
            print("TESTING FOR",test_name)
            # if test_name == train_name:
            #     continue
            test_csv = os.path.join(DATASETS_DIR, test_name, f"{test_name}.csv")
            windows_test = load_and_window(test_csv, WINDOW_SIZE)
            # Pick one random window index
            idx = 10
            w = windows_test[idx]

            start = time.time()
            is_anom, mse = is_window_anomaly(w, model, threshold)
            inference_time = time.time() - start

            results.append({
                "train_dataset": train_name,
                "test_dataset": test_name,
                "window_index": idx,
                "mse": mse,
                "is_anomaly": int(is_anom),
                "inference_time_sec": inference_time
            })

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(
        by=["train_dataset", "test_dataset", "window_index"]
    )
    df_sorted.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"Results saved to {RESULTS_CSV_PATH}")

if __name__ == "__main__":
    main()