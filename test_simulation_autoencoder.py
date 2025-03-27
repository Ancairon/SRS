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
# Synthetic Anomaly Generation Functions
# ============================
def inject_gaussian_noise(data: pd.DataFrame, mean: float = 0, std: float = 0.1) -> pd.DataFrame:
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + noise
    return noisy_data

def inject_uniform_noise(data: pd.DataFrame, low: float = -0.1, high: float = 0.1) -> pd.DataFrame:
    noise = np.random.uniform(low, high, size=data.shape)
    noisy_data = data + noise
    return noisy_data

def apply_salt_and_pepper_noise(data: pd.DataFrame, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> pd.DataFrame:
    noisy_data = data.copy()
    random_matrix = np.random.rand(*data.shape)
    for i, col in enumerate(data.columns):
        col_min = data[col].min()
        col_max = data[col].max()
        noisy_data.loc[random_matrix[:, i] < salt_prob, col] = col_max
        noisy_data.loc[random_matrix[:, i] > 1 - pepper_prob, col] = col_min
    return noisy_data

def mask_features_randomly(data: pd.DataFrame, mask_fraction: float = 0.1) -> pd.DataFrame:
    masked_data = data.copy()
    mask = np.random.rand(*data.shape) < mask_fraction
    masked_data[mask] = np.nan
    return masked_data

# ============================
# Domain-Shifted Data Generation Functions
# ============================
def simulate_sensor_drift(data: pd.DataFrame, drift_value: float = 0.5) -> pd.DataFrame:
    drifted_data = data + drift_value
    return drifted_data

def simulate_scaling_drift(data: pd.DataFrame, scale_factor: float = 1.1) -> pd.DataFrame:
    scaled_data = data * scale_factor
    return scaled_data

def simulate_time_based_variation(data: pd.DataFrame) -> pd.DataFrame:
    drift = np.linspace(0, 1, len(data))
    drifted_data = data.copy()
    for i in range(len(data)):
        drifted_data.iloc[i] = drifted_data.iloc[i] + drift[i]
    return drifted_data

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
    encoded = Dense(128, activation='relu', kernel_regularizer='l2')(input_layer)
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
    normal_errors = compute_reconstruction_error(autoencoder, normal_scaled)
    anomaly_errors = compute_reconstruction_error(autoencoder, anomaly_scaled)
    threshold = np.percentile(normal_errors, threshold_percentile)
    
    normal_preds = (normal_errors > threshold).astype(int)
    anomaly_preds = (anomaly_errors > threshold).astype(int)
    
    normal_labels = np.zeros(len(normal_errors), dtype=int)
    anomaly_labels = np.ones(len(anomaly_errors), dtype=int)
    
    y_true = np.concatenate([normal_labels, anomaly_labels])
    y_pred = np.concatenate([normal_preds, anomaly_preds])
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return f1, precision, recall, cm, threshold

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
    simulated_versions = {}
    
    # Define simulation functions in a dictionary
    simulation_functions = {
        'gaussian_noise': inject_gaussian_noise,
        'uniform_noise': inject_uniform_noise,
        'salt_pepper_noise': apply_salt_and_pepper_noise,
        'masked': mask_features_randomly,
        'sensor_drift': simulate_sensor_drift,
        'scaling_drift': simulate_scaling_drift,
        'time_variation': simulate_time_based_variation
    }
    
    # Train an autoencoder on each dataset and generate simulated anomaly versions
    for train_dataset in datasets:
        df = pd.read_csv(train_dataset)
        df = df.iloc[:, :-1]  # Drop target column
        train_scaled, scaler = preprocess_data(df)
        autoencoder, _ = build_autoencoder(train_scaled.shape[1])
        train_autoencoder(autoencoder, train_scaled)
        trained_models[train_dataset] = (autoencoder, train_scaled, scaler)
        
        # Generate simulated anomaly versions from the original data
        sim_versions = {}
        for sim_name, sim_func in simulation_functions.items():
            sim_df = sim_func(df)
            sim_versions[sim_name] = sim_df
        simulated_versions[train_dataset] = sim_versions
    
    # Evaluate: (a) original anomaly datasets from different machines
    for train_dataset, anomaly_dataset in sorted(product(datasets, datasets)):
        if train_dataset == anomaly_dataset:
            continue
        autoencoder, train_scaled, scaler = trained_models[train_dataset]
        normal_indices = np.random.choice(train_scaled.shape[0], size=1800, replace=False)
        normal_eval = train_scaled[normal_indices]
        
        anomaly_data = pd.read_csv(anomaly_dataset).sample(n=1800, random_state=SEED)
        anomaly_data = anomaly_data.iloc[:, :-1]
        anomaly_eval, _ = preprocess_data(anomaly_data, scaler)
        
        f1, precision, recall, cm, threshold = evaluate_balanced(autoencoder, normal_eval, anomaly_eval,threshold_arg)
        results.append({
            "Train Dataset": os.path.basename(train_dataset),
            "Anomaly Dataset": os.path.basename(anomaly_dataset),
            "Simulation": "original",
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": str(cm).replace("\\n", ""),
            "Threshold": threshold
        })
    
    # Evaluate: (b) simulated anomaly datasets (derived from the same dataset)
    for train_dataset in datasets:
        autoencoder, train_scaled, scaler = trained_models[train_dataset]
        normal_indices = np.random.choice(train_scaled.shape[0], size=1800, replace=False)
        normal_eval = train_scaled[normal_indices]
        sim_versions = simulated_versions[train_dataset]
        for sim_name, sim_df in sim_versions.items():
            sim_sample = sim_df.sample(n=1800, random_state=SEED)
            # In case masking introduced NaNs, fill them (e.g., with column means)
            sim_sample = sim_sample.fillna(sim_sample.mean())
            anomaly_eval, _ = preprocess_data(sim_sample, scaler)
            f1, precision, recall, cm, threshold = evaluate_balanced(autoencoder, normal_eval, anomaly_eval,threshold_arg)
            results.append({
                "Train Dataset": os.path.basename(train_dataset),
                "Anomaly Dataset": os.path.basename(train_dataset),
                "Simulation": sim_name,
                "F1-Score": f1,
                "Precision": precision,
                "Recall": recall,
                "Confusion Matrix": str(cm).replace("\\n", ""),
                "Threshold": threshold
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Combined evaluation results saved to {output_csv}")

    return results_df["F1-Score"].mean()

# Example usage
parent_dir = 'custom_datasets'  # Parent directory containing subdirectories for each dataset

averages = []

for i in range(5,100,5):
    print(i)
    output_csv = f'combined_evaluation_results_threshold_{i}.csv'

    averages.append({i:main(parent_dir, output_csv, i)})


for average in averages:
    print(average)
