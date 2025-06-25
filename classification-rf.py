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
        if not os.path.isfile(csv_path) or "idle" in label:
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

if __name__ == "__main__":
    main()
