import os
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def load_sequences(base_dir, seq_len, step, train_frac):
    X_tr, y_tr, X_te, y_te = [], [], [], []

    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        csv_path = os.path.join(label_dir, f"{label}.csv")
        print(label, "idle" not in label)
        if not os.path.isfile(csv_path) or "idle" in label:
            continue

        data = pd.read_csv(csv_path).drop(columns="Target")
        n_rows = data.shape[0]
        split = int(n_rows * train_frac)

        for start in range(0, split - seq_len + 1, step):
            X_tr.append(data[start:start+seq_len])
            y_tr.append(label)

        for start in range(split, n_rows - seq_len + 1, step):
            X_te.append(data[start:start+seq_len])
            y_te.append(label)
            break

    return (
        np.array(X_tr), np.array(y_tr),
        np.array(X_te), np.array(y_te)
    )

def main():
    base_dir   = "custom_datasets"
    seq_len    = 10
    step       = 10
    train_frac = .7

    X_train, y_train, X_test, y_test = load_sequences(base_dir, seq_len, step, train_frac)
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    le = LabelEncoder()
    y_train_enc = to_categorical(le.fit_transform(y_train))
    y_test_enc = to_categorical(le.transform(y_test))

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(y_train_enc.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start_train = time.time()
    model.fit(X_train, y_train_enc, epochs=10, batch_size=32, verbose=1)
    end_train = time.time()

    start_infer = time.time()
    y_pred_prob = model.predict(X_test)
    end_infer = time.time()

    y_pred = le.inverse_transform(np.argmax(y_pred_prob, axis=1))
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred,digits=3))
    print(f"Train time:    {end_train - start_train:.3f} seconds")
    print(f"Inference time:{end_infer - start_infer:.3f} seconds")

if __name__ == "__main__":
    main()
