import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_time_split_sequences(base_dir, seq_len, step, train_frac):
    X_tr, y_tr, X_te, y_te = [], [], [], []

    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        csv_path = os.path.join(label_dir, f"{label}.csv")
        if not os.path.isfile(csv_path) or "idle" in label or "redis" in label or "nginx" in label:
            continue

        data = pd.read_csv(csv_path).drop(columns="Target")
        n_rows = data.shape[0]
        split = int(n_rows * train_frac)

        for start in range(0, split - seq_len + 1, step):
            X_tr.append(data[start:start+seq_len].to_numpy().flatten())
            y_tr.append(label)

        for start in range(split, 50 + n_rows - seq_len + 1, step):
            X_te.append(data[start:start+seq_len].to_numpy().flatten())
            y_te.append(label)
            break

    return (
        np.vstack(X_tr), np.array(y_tr),
        np.vstack(X_te), np.array(y_te),
    )

def load_test_sequences(csv_path, seq_len, step):
    data = pd.read_csv(csv_path).drop(columns="Target")
    X_test = []
    # Take only one sequence, like in the original test
    start = 0
    if data.shape[0] >= seq_len:
        X_test.append(data[start:start+seq_len].to_numpy().flatten())
    return np.array(X_test)

def main():
    base_dir   = "custom_datasets"
    seq_len    = 3
    step       = 3
    train_frac = 0.7

    X_train, y_train, X_test, y_test = load_time_split_sequences(
        base_dir, seq_len, step, train_frac
    )
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    rf = RandomForestClassifier(random_state=42)
    start_train = time.time()
    rf.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train

    start_infer = time.time()
    y_pred = rf.predict(X_test)
    end_infer = time.time()
    infer_time = end_infer - start_infer

    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Train time:    {train_time:.3f} seconds")
    print(f"Inference time:{infer_time:.3f} seconds")

        # Now test on nginx dataset
    nginx = "custom_datasets/nginx/nginx.csv"
    if os.path.isfile(nginx):
        X_nginx = load_test_sequences(nginx, seq_len, step)
        print(f"\nNginx samples: {len(X_nginx)}")

        y_pred_nginx = rf.predict(X_nginx)
        y_proba_nginx = rf.predict_proba(X_nginx)

        print("\nPredictions on nginx:")
        for i, (pred, proba) in enumerate(zip(y_pred_nginx, y_proba_nginx)):
            conf_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(rf.classes_, proba)])
            print(f"Sample {i}: Predicted {pred}, Confidences: {{{conf_str}}}")
    else:
        print("nginx.csv not found")


    # Now test on redis dataset
    redis = "custom_datasets/redis/redis.csv"
    if os.path.isfile(redis):
        X_redis = load_test_sequences(redis, seq_len, step)
        print(f"\nRedis samples: {len(X_redis)}")

        y_pred_redis = rf.predict(X_redis)
        y_proba_redis = rf.predict_proba(X_redis)

        print("\nPredictions on redis:")
        for i, (pred, proba) in enumerate(zip(y_pred_redis, y_proba_redis)):
            conf_str = ", ".join([f"{c}: {p:.3f}" for c, p in zip(rf.classes_, proba)])
            print(f"Sample {i}: Predicted {pred}, Confidences: {{{conf_str}}}")
    else:
        print("redis.csv not found")


if __name__ == "__main__":
    main()