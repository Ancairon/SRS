import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import random
import os
from itertools import product
import time

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Load datasets


def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Drop the target column (last column)
    train_data = train_data.iloc[:, :-1]
    test_data = test_data.iloc[:, :-1]

    return train_data, test_data

# Preprocess datasets


def preprocess_data(train_data, test_data):
    # Convert to numpy arrays
    train_array = train_data.to_numpy()
    test_array = test_data.to_numpy()

    # Normalize test data using training statistics
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_array)
    test_scaled = scaler.transform(test_array)

    return train_scaled, test_scaled, scaler

# Build Autoencoder model


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu', kernel_regularizer='l2')(
        input_layer)  # Increased layer size and added L2 regularization
    encoded = Dropout(0.2)(encoded)  # Added dropout for regularization
    encoded = Dense(64, activation='relu', kernel_regularizer='l2')(encoded)
    latent = Dense(32, activation='relu', name='latent_space')(
        encoded)  # Increased latent space size
    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=latent)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Train autoencoder


def train_autoencoder(autoencoder, train_data, epochs=50, batch_size=32):
    history = autoencoder.fit(train_data, train_data,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_split=0.2,
                              verbose=0)  # Suppress training output
    return history

# Compute reconstruction error


def compute_reconstruction_error(autoencoder, data):
    reconstructed = autoencoder.predict(data)
    mse = np.mean(np.power(data - reconstructed, 2), axis=1)
    return mse

# Main pipeline for dataset combination


def evaluate_datasets(autoencoder, train_scaled, test_scaled):
    # Load and preprocess test data

    # Compute reconstruction errors for training and test data
    train_errors = compute_reconstruction_error(autoencoder, train_scaled)

    # Threshold for anomaly detection (based on training errors)
    # Dynamic threshold based on mean and std
    threshold = train_errors.mean() + 2 * train_errors.std()

    # Measure inference time for test data
    start_time = time.time()
    test_errors = compute_reconstruction_error(autoencoder, test_scaled)
    inference_time = time.time() - start_time

    # Predictions (only for test data)
    test_predictions = test_errors > threshold

    # Labels: all samples in the test set are considered anomalies
    # Test data is assumed out-of-distribution
    y_true = np.ones(len(test_errors))
    y_pred = test_predictions

    # Metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(x) for x in cm.ravel()
                      ) if cm.size == 4 else (0, 0, 0, int(cm[0]))

    print(f"Threshold: {threshold}")
    print(
        f"Test Errors (min, max, mean): {test_errors.min()}, {test_errors.max()}, {test_errors.mean()}")
    print(
        f"Test Predictions: {test_predictions.sum()} positives out of {len(test_predictions)}")

    return f1, precision, recall, tn, fp, fn, tp, inference_time

# Iterate over dataset combinations and save results


def main(parent_dir, output_csv):
    datasets = [os.path.join(parent_dir, d, f"{d}.csv") for d in os.listdir(
        parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    results = []

    trained_models = {}

    for train_dataset in datasets:
        train_data = pd.read_csv(train_dataset)
        train_data = train_data.iloc[:, :-1]  # Drop target column
        train_scaled, _, scaler = preprocess_data(train_data, train_data)

        # Train model once per train dataset
        autoencoder, _ = build_autoencoder(train_scaled.shape[1])
        train_autoencoder(autoencoder, train_scaled)
        trained_models[train_dataset] = (autoencoder, train_scaled, scaler)

    print(sorted(product(datasets, datasets)))

    for train_dataset, test_dataset in sorted(product(datasets, datasets)):

        autoencoder, train_scaled, scaler = trained_models[train_dataset]
        test_data = pd.read_csv(test_dataset).sample(n=1)
        test_data = test_data.iloc[:, :-1]  # Drop target column
        # Normalize test data using training scaler
        test_scaled = scaler.transform(test_data.to_numpy())

        f1, precision, recall, tn, fp, fn, tp, inference_time = evaluate_datasets(
            autoencoder, train_scaled, test_scaled)

        results.append({
            "Train Dataset": os.path.basename(train_dataset),
            "Test Dataset": os.path.basename(test_dataset),
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Inference Time (s)": inference_time
        })

        print({
            "Train Dataset": os.path.basename(train_dataset),
            "Test Dataset": os.path.basename(test_dataset),
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Inference Time (s)": inference_time
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# Example usage
parent_dir = 'custom_datasets'  # Replace with your parent directory
output_csv = 'evaluation_results.csv'           # Output CSV path
main(parent_dir, output_csv)
