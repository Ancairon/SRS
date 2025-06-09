import os
import numpy as np
import pandas as pd
import time
import random
import warnings

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# use SARIMAX instead of the newer ARIMA class
from statsmodels.tsa.statespace.sarimax import SARIMAX

# suppress convergence warnings
warnings.filterwarnings("ignore")

# ============================
# Set seeds & determinism
# ============================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================
# Data Loading and Preprocessing
# ============================
def load_data(train_path, test_path):
    train = pd.read_csv(train_path).iloc[:, :-1]
    test  = pd.read_csv(test_path).iloc[:, :-1]
    return train, test

def preprocess_data(df, scaler=None):
    arr = df.values
    if scaler is None:
        scaler = StandardScaler().fit(arr)
    scaled = scaler.transform(arr)
    return scaled, scaler

# ============================
# ARIMA–Based Anomaly Detector
# ============================
def build_arma_detectors(train_scaled, order=(5,1,0)):
    """
    Fit one SARIMAX(order) model per feature (column).
    """
    n_features = train_scaled.shape[1]
    models = []
    for j in range(n_features):
        series = train_scaled[:, j]
        mdl = SARIMAX(series,
                      order=order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        res = mdl.fit(disp=False)
        models.append(res)
    return models

def compute_anomaly_scores_arma(models, train_scaled, eval_scaled):
    """
    Forecast len(eval_scaled) steps ahead for each feature,
    then compute mean absolute error across features.
    """
    horizon = eval_scaled.shape[0]

    # collect residuals from training for thresholding
    all_resids = np.hstack([np.abs(m.resid) for m in models])

    # forecast each series
    forecasts = np.vstack([
        m.forecast(steps=horizon) for m in models
    ]).T  # shape (horizon, n_features)

    errors = np.mean(np.abs(eval_scaled - forecasts), axis=1)
    return errors, all_resids

# ============================
# Balanced Evaluation Function
# ============================
def evaluate_balanced_arma(models, train_scaled, normal_scaled, anomaly_scaled, 
                           threshold_percentile=80):
    start = time.time()

    norm_errs, resid    = compute_anomaly_scores_arma(models, train_scaled, normal_scaled)
    anom_errs, _        = compute_anomaly_scores_arma(models, train_scaled, anomaly_scaled)

    threshold = np.percentile(resid, threshold_percentile)

    normal_preds  = (norm_errs  > threshold).astype(int)
    anomaly_preds = (anom_errs  > threshold).astype(int)

    y_true = np.concatenate([np.zeros_like(normal_preds), np.ones_like(anomaly_preds)])
    y_pred = np.concatenate([normal_preds, anomaly_preds])

    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    inference_time = time.time() - start

    return f1, precision, recall, tn, fp, fn, tp, threshold, inference_time

# ============================
# Main Pipeline with ARIMA
# ============================
def main_arma(parent_dir, output_csv, arma_order=(5,1,0)):
    datasets = [
        os.path.join(parent_dir, d, f"{d}.csv")
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    trained = {}
    for train_path in datasets:
        df = pd.read_csv(train_path).iloc[:, :-1]
        scaled, scaler = preprocess_data(df)
        models = build_arma_detectors(scaled, order=arma_order)
        trained[train_path] = (models, scaler, scaled)

    results = []
    for train_path in datasets:
        for anomaly_path in datasets:
            if "rpi" in train_path or "rpi" in anomaly_path:
                continue

            models, scaler, train_scaled = trained[train_path]

            # single normal point
            idx = np.random.choice(train_scaled.shape[0], 1, replace=False)
            normal_eval = train_scaled[idx]

            # 1 anomaly samples
            df_anom = pd.read_csv(anomaly_path).sample(n=1, random_state=SEED).iloc[:, :-1]
            anomaly_eval, _ = preprocess_data(df_anom, scaler)

            f1, prec, rec, tn, fp, fn, tp, thr, t_inf = evaluate_balanced_arma(
                models, train_scaled, normal_eval, anomaly_eval
            )

            results.append({
                "Train Dataset":   os.path.basename(train_path),
                "Anomaly Dataset": os.path.basename(anomaly_path),
                "tn":              tn,
                "tp":              tp,
                "fn":              fn,
                "fp":              fp,
            })
    print(np.average(t_inf))

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"ARIMA-based evaluation results saved to {output_csv}")

if __name__ == "__main__":
    main_arma('custom_datasets', '1sample-arima_evaluation_results.csv', arma_order=(5,1,0))
