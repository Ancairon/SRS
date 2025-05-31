import os
import numpy as np
import pandas as pd
import time
import random
from itertools import product

from sklearn.ensemble      import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    f1_score, precision_score, recall_score, confusion_matrix
)

# ─── reproducibility ─────────────────────────────────────────────
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─── helper functions ────────────────────────────────────────────
def load_csv_data(path):
    # drop last column if it's a target
    df = pd.read_csv(path)
    return df.iloc[:, :-1]

def preprocess_data(df, scaler=None):
    arr = df.to_numpy()
    if scaler is None:
        scaler = StandardScaler().fit(arr)
    return scaler.transform(arr), scaler

def make_sequences(arr, seq_len=10, step=10):
    """
    Turn a T×F array into N×(seq_len*F) by 
    flattening non-overlapping windows of length seq_len.
    """
    seqs = []
    T, F = arr.shape
    for start in range(0, T - seq_len + 1, step):
        window = arr[start : start + seq_len]
        seqs.append(window.flatten())
    return np.array(seqs)

def build_rf_detector(X_train):
    # now X_train is shape (n_windows, seq_len*F)
    model = IsolationForest(random_state=SEED, contamination='auto')
    model.fit(X_train)
    return model

def compute_anomaly_score(model, X):
    # X is shape (n_samples, seq_len*F), returns one score per sample
    return -model.score_samples(X)

def evaluate_balanced_rf(model, normal_X, anomaly_X, threshold_pct=80):
    t0 = time.time()
    scores_norm = compute_anomaly_score(model, normal_X)
    scores_anom = compute_anomaly_score(model, anomaly_X)
    threshold   = np.percentile(scores_norm, threshold_pct)

    preds_norm = (scores_norm > threshold).astype(int)
    preds_anom = (scores_anom > threshold).astype(int)
    t1 = time.time()

    y_true = np.concatenate([
        np.zeros(len(scores_norm), dtype=int),
        np.ones(len(scores_anom), dtype=int)
    ])
    y_pred = np.concatenate([preds_norm, preds_anom])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (
        f1_score(y_true, y_pred, zero_division=0),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        tn, fp, fn, tp,
        threshold,
        t1 - t0
    )

# ─── main pipeline ───────────────────────────────────────────────
def main_rf_sequences(parent_dir, output_csv,
                      seq_len=10, step=10, threshold_pct=80):
    # 1) find all <dir>/<dir>.csv
    datasets = [
        os.path.join(parent_dir, d, f"{d}.csv")
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]

    # 2) train one IF detector per dataset on its 10-row sequences
    trained = {}
    for path in datasets:
        df = load_csv_data(path)
        scaled, scaler = preprocess_data(df)
        seqs = make_sequences(scaled, seq_len, step)
        iso = build_rf_detector(seqs) 
        trained[path] = (iso, seqs, scaler)

    results = []
    # 3) cross-evaluate: for each train vs. anomaly pair
    for train_path, anom_path in product(datasets, datasets):
        if train_path == anom_path:
            continue

        iso, train_seqs, scaler = trained[train_path]

        # pick one random "normal" window
        idx = random.randrange(train_seqs.shape[0])
        normal_eval = train_seqs[idx : idx + 1]  # shape (1, seq_len*F)

        # build anomaly windows from the other CSV
        df_anom = load_csv_data(anom_path)
        anom_scaled, _ = preprocess_data(df_anom, scaler)
        anom_seqs     = make_sequences(anom_scaled, seq_len, step)

        # sample up to 10 anomaly windows
        n_eval = min(1, anom_seqs.shape[0])
        choices = random.sample(range(anom_seqs.shape[0]), k=n_eval)
        anomaly_eval = anom_seqs[choices]

        # get metrics
        f1, prec, rec, tn, fp, fn, tp, thr, inf_time = evaluate_balanced_rf(
            iso, normal_eval, anomaly_eval, threshold_pct=threshold_pct
        )
        results.append({
            "Train Dataset":   os.path.basename(train_path),
            "Anomaly Dataset": os.path.basename(anom_path),
            "Inference time":  inf_time,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Threshold": thr
        })

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    main_rf_sequences('custom_datasets', 'rf_seq_results.csv')
