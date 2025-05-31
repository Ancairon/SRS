import os
import glob
import numpy as np
import pandas as pd

from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import classification_report, accuracy_score

def load_time_split_sequences(base_dir, seq_len=10, step=5, train_frac=0.7):
    """
    For each CSV under base_dir/<label>/<label>.csv:
      - take the first train_frac of its rows → train snippets
      - the last (1-train_frac) of its rows → test snippets
    Extract overlapping seqs of length seq_len (step apart).

    Returns:
      X_train, y_train, X_test, y_test
    """
    X_tr, y_tr = [], []
    X_te, y_te = [], []

    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        if not os.path.isdir(label_dir):
            continue

        # assume exactly one file named <label>.csv per folder
        csv_path = os.path.join(label_dir, f"{label}.csv")
        if not os.path.isfile(csv_path):
            continue

        print(f"→ Loading {csv_path!r} as label '{label}'")
        data = pd.read_csv(csv_path).values
        n_rows, n_feats = data.shape
        split_row = int(n_rows * train_frac)

        # TRAIN snippets: from rows [0 .. split_row)
        for start in range(0, split_row - seq_len + 1, step):
            seq = data[start : start + seq_len]
            X_tr.append(seq.flatten())
            y_tr.append(label)

        # TEST snippets: from rows [split_row .. end)
        for start in range(split_row, n_rows - seq_len + 1, step):
            seq = data[start : start + seq_len]
            X_te.append(seq.flatten())
            y_te.append(label)

    X_train = np.vstack(X_tr)
    y_train = np.array(y_tr)
    X_test  = np.vstack(X_te)
    y_test  = np.array(y_te)

    return X_train, y_train, X_test, y_test


def main():
    base_dir = "custom_datasets"
    seq_len  = 10
    step     = 5
    train_frac = 0.7

    # 1) load + time‐split
    X_train, y_train, X_test, y_test = load_time_split_sequences(
        base_dir, seq_len, step, train_frac
    )
    print(f"\nSamples →  Train: {len(y_train)},  Test: {len(y_test)}")
    print(f"Feature dimensionality: {X_train.shape[1]}\n")

    # 2) scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 3) hyperparameter search on TRAIN
    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="accuracy",
        verbose=2,
    )
    gs.fit(X_train, y_train)
    print("Best hyperparameters:", gs.best_params_, "\n")

    # 4) evaluate on the truly unseen snippets
    best_rf = gs.best_estimator_
    y_pred  = best_rf.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    print(f"Test accuracy on held‐out snippets: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
