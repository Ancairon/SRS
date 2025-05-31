import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_time_split_sequences(base_dir, seq_len, step, train_frac):
    """
    For each CSV under base_dir/<label>/<label>.csv:
      - first train_frac of rows → train snippets
      - remaining rows → test snippets
    Extract overlapping windows of length seq_len (step apart).
    """
    X_tr, y_tr, X_te, y_te = [], [], [], []

    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        csv_path = os.path.join(label_dir, f"{label}.csv")
        if not os.path.isfile(csv_path):
            continue

        data = pd.read_csv(csv_path).values
        n_rows = data.shape[0]
        split = int(n_rows * train_frac)

        # train windows
        for start in range(0, split - seq_len + 1, step):
            X_tr.append(data[start:start+seq_len].flatten())
            y_tr.append(label)

        # test windows
        for start in range(split, n_rows - seq_len + 1, step):
            X_te.append(data[start:start+seq_len].flatten())
            y_te.append(label)

    return (
        np.vstack(X_tr), np.array(y_tr),
        np.vstack(X_te), np.array(y_te),
    )

def main():
    base_dir   = "custom_datasets"
    seq_len    = 10
    step       = 10
    train_frac = .05

    # load & split
    X_train, y_train, X_test, y_test = load_time_split_sequences(
        base_dir, seq_len, step, train_frac
    )
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # train RF
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # evaluate
    y_pred = rf.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
