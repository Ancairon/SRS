import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import random
import time
from itertools import product

# ============================
# Set seeds & determinism
# ============================
SEED = 42
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================
# Data Loading and Preprocessing Functions
# ============================


def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data = train_data.iloc[:, :-1]
    test_data = test_data.iloc[:, :-1]
    return train_data, test_data


def preprocess_data(data, scaler=None):
    data_array = data.to_numpy()
    if scaler is None:
        scaler = StandardScaler().fit(data_array)
    data_scaled = scaler.transform(data_array)
    return data_scaled, scaler

# ============================
# Autoencoder Model and Training
# ============================


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu',
                    kernel_regularizer='l2')(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='relu', kernel_regularizer='l2')(encoded)
    latent = Dense(32, activation='relu', name='latent_space')(encoded)
    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=latent)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def train_autoencoder(autoencoder, train_data, epochs=50, batch_size=32):
    history = autoencoder.fit(train_data, train_data,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=0)
    return history


def compute_reconstruction_error(autoencoder, data):
    reconstructed = autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructed, 2), axis=1)
    return mse

# ============================
# Balanced Evaluation Function
# ============================


def evaluate_balanced(autoencoder, normal_scaled, anomaly_scaled, threshold_percentile):
    start = time.time()
    normal_errors = compute_reconstruction_error(autoencoder, normal_scaled)
    anomaly_errors = compute_reconstruction_error(autoencoder, anomaly_scaled)
    threshold = np.percentile(normal_errors, threshold_percentile)

    normal_preds = (normal_errors > threshold).astype(int)
    anomaly_preds = (anomaly_errors > threshold).astype(int)

    end = time.time()
    inference_time = (end - start)

    normal_labels = np.zeros(len(normal_errors), dtype=int)
    anomaly_labels = np.ones(len(anomaly_errors), dtype=int)

    y_true = np.concatenate([normal_labels, anomaly_labels])
    y_pred = np.concatenate([normal_preds, anomaly_preds])

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return f1, precision, recall, tn, fp, fn, tp, threshold, inference_time

# ============================
# Main Pipeline
# ============================


def main(parent_dir, output_csv, threshold_arg):

    # Gather dataset CSV paths from subdirectories
    datasets = [os.path.join(parent_dir, d, f"{d}.csv")
                for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d))]

    results = []
    trained_models = {}

    # Train an autoencoder on each dataset and generate simulated anomaly versions
    for train_dataset in datasets:
        df = pd.read_csv(train_dataset)
        df = df.iloc[:, :-1]  # Drop target column
        train_scaled, scaler = preprocess_data(df)
        autoencoder, _ = build_autoencoder(train_scaled.shape[1])
        train_autoencoder(autoencoder, train_scaled)
        trained_models[train_dataset] = (autoencoder, train_scaled, scaler)

        # # Generate simulated anomaly versions from the original data
        # sim_versions = {}
        # for sim_name, sim_func in simulation_functions.items():
        #     sim_df = sim_func(df)
        #     sim_versions[sim_name] = sim_df
        # simulated_versions[train_dataset] = sim_versions

    # Evaluate: (a) original anomaly datasets from different machines
    for train_dataset, anomaly_dataset in sorted(product(datasets, datasets)):
        # if "rpi" in train_dataset or "rpi" in anomaly_dataset or train_dataset == anomaly_dataset:
        if "rpi" in train_dataset or "rpi" in anomaly_dataset:
            continue
        autoencoder, train_scaled, scaler = trained_models[train_dataset]
        normal_indices = np.random.choice(
            train_scaled.shape[0], size=1, replace=False)
        normal_eval = train_scaled[normal_indices]

        anomaly_data = pd.read_csv(anomaly_dataset).sample(
            n=20, random_state=SEED)
        anomaly_data = anomaly_data.iloc[:, :-1]
        anomaly_eval, _ = preprocess_data(anomaly_data, scaler)

        f1, precision, recall, tn, fp, fn, tp, threshold, inference_time = evaluate_balanced(
            autoencoder, normal_eval, anomaly_eval, threshold_arg)
        results.append({
            "Train Dataset": os.path.basename(train_dataset),
            "Anomaly Dataset": os.path.basename(anomaly_dataset),
            "Inference time": inference_time,
            "tn": tn,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Threshold": threshold
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Combined evaluation results saved to {output_csv}")

    return results_df["F1-Score"].mean()


# Example usage
# Parent directory containing subdirectories for each dataset
parent_dir = 'custom_datasets'

averages = []

# for i in range(10, 100, 10):
#     print(i)
#     output_csv = f'evaluation_results_ae_threshold_{i}.csv'

#     averages.append({i: main(parent_dir, output_csv, i)})

output_csv = f'evaluation_results_ae_threshold_{80}.csv'

averages.append({80: main(parent_dir, output_csv, 80)})


for average in averages:
    print(average)
