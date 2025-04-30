import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import random
import time
from itertools import product

# ============================
# Set seeds & determinism
# ============================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
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
# Random Forest–Based Anomaly Detector (IsolationForest)
# ============================


def build_rf_detector(train_data):
    """
    Trains an IsolationForest on the provided normal (training) data.
    """
    # Using IsolationForest (a tree-based anomaly detector)
    model = IsolationForest(random_state=SEED, contamination='auto')
    model.fit(train_data)
    return model


def compute_anomaly_score(model, data):
    """
    Computes an anomaly score for each sample.
    We take the negative of score_samples so that higher values indicate more anomalous behavior.
    """
    # Note: IsolationForest.score_samples returns higher values for more normal points.
    # Taking the negative inverts that.
    scores = -model.score_samples(data)
    return scores

# ============================
# Balanced Evaluation Function for RF Detector
# ============================


def evaluate_balanced_rf(model, normal_scaled, anomaly_scaled, threshold_percentile=80):
    start = time.time()
    normal_scores = compute_anomaly_score(model, normal_scaled)
    anomaly_scores = compute_anomaly_score(model, anomaly_scaled)

    # Set threshold based on the normal scores (e.g., 95th percentile)
    threshold = np.percentile(normal_scores, threshold_percentile)

    # Prediction: if score > threshold, classify as anomaly (1), else normal (0)
    normal_preds = (normal_scores > threshold).astype(int)
    anomaly_preds = (anomaly_scores > threshold).astype(int)

    end = time.time()
    inference_time = (end - start)

    normal_labels = np.zeros(len(normal_scores), dtype=int)
    anomaly_labels = np.ones(len(anomaly_scores), dtype=int)

    y_true = np.concatenate([normal_labels, anomaly_labels])
    y_pred = np.concatenate([normal_preds, anomaly_preds])

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return f1, precision, recall, tn, fp, fn, tp, threshold, inference_time

# ============================
# Main Pipeline for Combined Evaluation with RF
# ============================


def main_rf(parent_dir, output_csv):
    # Gather dataset CSV paths from subdirectories
    datasets = [os.path.join(parent_dir, d, f"{d}.csv")
                for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d))]

    results = []
    trained_models = {}

    # Train an RF-based anomaly detector on each dataset and generate simulated versions
    for train_dataset in datasets:
        df = pd.read_csv(train_dataset)
        df = df.iloc[:, :-1]  # Drop target column
        train_scaled, scaler = preprocess_data(df)
        rf_model = build_rf_detector(train_scaled)
        trained_models[train_dataset] = (rf_model, train_scaled, scaler)

    # Evaluation (a): Use original anomaly datasets from different machines
    for train_dataset, anomaly_dataset in sorted(product(datasets, datasets)):
        if "rpi" in train_dataset or "rpi" in anomaly_dataset:
            continue
        rf_model, train_scaled, scaler = trained_models[train_dataset]
        normal_indices = np.random.choice(
            train_scaled.shape[0], size=1, replace=False)
        normal_eval = train_scaled[normal_indices]

        anomaly_data = pd.read_csv(anomaly_dataset).sample(
            n=30, random_state=SEED)
        anomaly_data = anomaly_data.iloc[:, :-1]
        anomaly_eval, _ = preprocess_data(anomaly_data, scaler)

        f1, precision, recall, tn, fp, fn, tp, threshold, inference_time = evaluate_balanced_rf(
            rf_model, normal_eval, anomaly_eval)
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
    print(f"RF-based evaluation results saved to {output_csv}")


# Example usage:
parent_dir = 'custom_datasets'  # Directory with subdirectories for each dataset
output_csv = '30sample-rf_evaluation_results.csv'
main_rf(parent_dir, output_csv)
