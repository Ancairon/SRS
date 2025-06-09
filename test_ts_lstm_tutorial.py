import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import glob
import os

# ------------------ Configuration ------------------ #
TRAIN_CSV_PATH   = 'custom_datasets/audio_classification/audio_classification.csv'
TRAIN_TARGET_COL = 'Target'
TEST_CSV_PATH    = 'custom_datasets/face_detection/face_detection.csv'
TEST_TARGET_COL  = 'Target'

WINDOW_SIZE   = 10     # seconds per window
STRIDE        = 10     # sliding-window stride
RANDOM_SEED   = 42
NUM_EPOCHS    = 20
LR            = 1e-3
EMBEDDING_DIM = 128

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Data Loading ------------------ #
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test  = pd.read_csv(TEST_CSV_PATH).sample(10)

# Drop target column, assume only normal samples
train_vals = df_train.drop(TRAIN_TARGET_COL, axis=1).to_numpy()  # (N_train, n_features)
test_vals  = df_test.drop(TEST_TARGET_COL,  axis=1).to_numpy()  # (N_test, n_features)

# ------------------ Windowing Helper ------------------ #
def make_windows(arr: np.ndarray, window_size: int, stride: int):
    sw = sliding_window_view(arr, (window_size, arr.shape[1]))
    sw = sw[:, 0, :, :]  # (num_windows, window_size, n_features)
    return sw[::stride]

# Build sliding windows
train_wins = make_windows(train_vals, WINDOW_SIZE, STRIDE)
test_wins  = make_windows(test_vals,  WINDOW_SIZE, STRIDE)

# Split train into train/val
train_arr, val_arr = train_test_split(train_wins, test_size=0.15, random_state=RANDOM_SEED)

# ------------------ Dataset helper ------------------ #
def create_dataset(arr3d: np.ndarray):
    seq_len    = arr3d.shape[1]
    n_features = arr3d.shape[2]
    dataset = [torch.tensor(window, dtype=torch.float32).to(device) for window in arr3d]
    return dataset, seq_len, n_features

# Create datasets
train_dataset, seq_len, n_features = create_dataset(train_arr)
val_dataset,   _,       _          = create_dataset(val_arr)
test_dataset,  _,       _          = create_dataset(test_wins)

# ------------------ Model Definition ------------------ #
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.rnn1 = nn.LSTM(n_features, 2*embedding_dim, batch_first=True)
        self.rnn2 = nn.LSTM(2*embedding_dim, embedding_dim, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(0)                     # (1, seq_len, n_features)
        x, _ = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        # hidden_n: (num_layers, batch, embedding_dim)
        embedding = hidden_n.squeeze(0).squeeze(0)  # -> (embedding_dim,)
        return embedding

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.rnn1       = nn.LSTM(embedding_dim, 2*embedding_dim, batch_first=True)
        self.rnn2       = nn.LSTM(2*embedding_dim, n_features, batch_first=True)

    def forward(self, embed):
        x = embed.unsqueeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)  # (1, seq_len, embedding_dim)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return x  # (1, seq_len, n_features)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, n_features, embedding_dim)

    def forward(self, x):
        embed = self.encoder(x)
        return self.decoder(embed)

# Instantiate model
model = RecurrentAutoencoder(seq_len, n_features, EMBEDDING_DIM).to(device)

# ------------------ Training Function ------------------ #
def train_model(model, train_dataset, val_dataset, n_epochs, device, criterion, optimizer):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, n_epochs+1):
        model.train(); train_losses = []
        for seq_true in train_dataset:
            seq_true = seq_true.to(device)
            optimizer.zero_grad()
            seq_pred = model(seq_true).squeeze(0)
            loss = criterion(seq_pred, seq_true)
            loss.backward(); optimizer.step()
            train_losses.append(loss.item())

        model.eval(); val_losses = []
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true).squeeze(0)
                val_losses.append(criterion(seq_pred, seq_true).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # print(f"Epoch {epoch}/{n_epochs} — train: {train_loss:.4f}, val: {val_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model.eval(), history

# Loss and optimizer
criterion = nn.L1Loss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=LR)

results = []
dataset_paths = []
PARENT_DIR = "custom_datasets"
for folder in sorted(os.listdir(PARENT_DIR)):
    folder_dir = os.path.join(PARENT_DIR, folder)
    csv_path = os.path.join(folder_dir, f"{folder}.csv")
    if os.path.isdir(folder_dir) and os.path.isfile(csv_path):
        dataset_paths.append(csv_path)

if not dataset_paths:
    raise FileNotFoundError(
        f"No CSV files found in {PARENT_DIR!r} with the expected structure.")

for train_path in dataset_paths:
    train_name = os.path.basename(os.path.dirname(train_path))
    print("TRAIN",train_name)

    # --- Prepare train/val for this dataset ---
    df_train   = pd.read_csv(train_path)
    train_vals = df_train.drop(TRAIN_TARGET_COL, axis=1).to_numpy()
    train_wins = make_windows(train_vals, WINDOW_SIZE, STRIDE)
    train_arr, val_arr = train_test_split(train_wins, test_size=0.15, random_state=RANDOM_SEED)
    train_dataset, seq_len, n_features = create_dataset(train_arr)
    val_dataset,   _,       _          = create_dataset(val_arr)

    # --- (Re)instantiate & train model ---
    model     = RecurrentAutoencoder(seq_len, n_features, EMBEDDING_DIM).to(device)
    criterion = nn.L1Loss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model, history = train_model(
        model, train_dataset, val_dataset,
        NUM_EPOCHS, device,
        criterion, optimizer
    )

    # --- Compute threshold on validation set ---
    with torch.no_grad():
        val_losses = [
            criterion(model(seq).squeeze(0), seq).item()
            for seq in val_dataset
        ]
    threshold = float(np.percentile(val_losses, 95))

    # --- Test on every other dataset ---
    for test_path in dataset_paths:
        # if test_path == train_path:
        #     continue
        test_name = os.path.basename(os.path.dirname(test_path))
        print("TEST",test_name)
        df_test   = pd.read_csv(test_path).sample(10)
        test_vals = df_test.drop(TEST_TARGET_COL, axis=1).to_numpy()
        test_wins = make_windows(test_vals, WINDOW_SIZE, STRIDE)
        test_dataset, _, _ = create_dataset(test_wins)

        with torch.no_grad():
            for idx, seq in enumerate(test_dataset):
                seq     = seq.to(device)
                pred    = model(seq).squeeze(0)
                loss    = criterion(pred, seq).item()
                is_anom = loss > threshold

                results.append({
                    'train_dataset': train_name,
                    'test_dataset':  test_name,
                    'window_index':  idx,
                    'loss':          loss,
                    'anomaly':       is_anom
                })

# Save the full cross‐dataset results
pd.DataFrame(results).to_csv('cross_dataset_results.csv', index=False)