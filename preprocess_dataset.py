import os
import glob
import numpy as np
import pandas as pd
import json  # Add json import
from sklearn.model_selection import train_test_split
from absl import app, flags
import pickle as pkl

FLAGS = flags.FLAGS

def load_mv_ucr_data_arff(dataset_name, parent_file):
    # Extract Data Dimensions from CSV
    dim_df = pd.read_csv("DataDimensions.csv")
    ds_idx = dim_df[dim_df["Problem"] == dataset_name].index[0]
    ds_trn_size = int(dim_df.at[ds_idx, "TrainSize"])
    ds_tst_size = int(dim_df.at[ds_idx, "TestSize"])
    ds_channel_nb = int(dim_df.at[ds_idx, "NumDimensions"])
    ds_seg_size = int(dim_df.at[ds_idx, "SeriesLength"])

    # Extract Train Data
    X_train = np.zeros((ds_trn_size, 1, ds_seg_size, ds_channel_nb))
    for ch in range(ds_channel_nb):
        if ds_channel_nb > 1:
            data_file = parent_file + "/" + dataset_name + "Dimension" + str(ch + 1) + "_TRAIN.arff"
        else:
            data_file = parent_file + "/" + dataset_name + "_TRAIN.arff"
        data, meta = arff.loadarff(data_file)
        train_data = data[meta.names()[:-1]]  # Everything but the last column
        train_data = np.array(train_data.tolist())
        X_train[:, :, :, ch] = train_data.reshape((ds_trn_size, 1, ds_seg_size))

    # Extract Train Labels
    data, meta = arff.loadarff(open(data_file, "r"))
    train_lbl = data[meta.names()[-1]]  # Last Column
    train_lbl = np.array([ss.decode('ascii') for ss in train_lbl])
    labels = {y: i for i, y in enumerate(np.sort(np.unique(train_lbl)))}
    y_train = np.array([labels[y] for y in train_lbl])

    # Extract Test Data
    X_test = np.zeros((ds_tst_size, 1, ds_seg_size, ds_channel_nb))
    for ch in range(ds_channel_nb):
        if ds_channel_nb > 1:
            data_file = parent_file + "/" + dataset_name + "Dimension" + str(ch + 1) + "_TEST.arff"
        else:
            data_file = parent_file + "/" + dataset_name + "_TEST.arff"
        data, meta = arff.loadarff(data_file)
        test_data = data[meta.names()[:-1]]  # Everything but the last column
        test_data = np.array(test_data.tolist())
        X_test[:, :, :, ch] = test_data.reshape((ds_tst_size, 1, ds_seg_size))

    # Extract Test Labels
    data, meta = arff.loadarff(open(data_file, "r"))
    test_lbl = data[meta.names()[-1]]  # Last Column
    test_lbl = np.array([ss.decode('ascii') for ss in test_lbl])
    y_test = np.array([labels[y] for y in test_lbl])

    # Shuffle Train and Test Data
    rand_indices = np.arange(X_train.shape[0])
    np.random.shuffle(rand_indices)
    X_train = X_train[rand_indices]
    y_train = y_train[rand_indices]
    rand_indices = np.arange(X_test.shape[0])
    np.random.shuffle(rand_indices)
    X_test = X_test[rand_indices]
    y_test = y_test[rand_indices]

    return X_train, y_train, X_test, y_test

def load_mv_ucr_data_csv(parent_file, dataset_name, test_size=0.2):
    # Extract Data Dimensions
    dim_df = pd.read_csv("DataDimensions.csv")
    ds_idx = dim_df[dim_df["Problem"] == dataset_name].index[0]
    ds_trn_size = int(dim_df.at[ds_idx, "TrainSize"])
    ds_channel_nb = int(dim_df.at[ds_idx, "NumDimensions"])
    ds_seg_size = int(dim_df.at[ds_idx, "SeriesLength"])

    # Find all CSV files with the dataset name as prefix
    csv_files = sorted(glob.glob(os.path.join(parent_file, f"{dataset_name}*.csv")))

    if len(csv_files) != ds_channel_nb:
        raise ValueError(f"Expected {ds_channel_nb} dimensions, but found {len(csv_files)} CSV files.")

    # Initialize list to store data from the last column of all CSVs
    all_data = []

    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        # Ensure that each CSV has 3600 rows as expected
        if data.shape[0] != ds_seg_size:
            raise ValueError(f"CSV {csv_file} does not have 3600 rows, found {data.shape[0]} rows instead.")
        
        # Extract only the last column from each CSV
        last_column_data = data.iloc[:, -1].values
        all_data.append(last_column_data)

    # Stack data as separate columns (dimensions)
    X = np.column_stack(all_data)

    # Check if the data is 2D (samples, dimensions), reshape if needed
    # Note that X.shape[1] will be the number of dimensions (CSV files)
    if len(X.shape) == 2:
        X = X.reshape((X.shape[0], 1, X.shape[1]))  # Shape to (samples, 1, dimensions)

    # The number of samples is the number of rows in the dataset (each row is a time point)
    n_samples = X.shape[0]

    # Add a static target column with the value 1 for every sample
    y = np.zeros(n_samples, dtype=int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test

def main(argv):
    dataset_name = FLAGS.dataset_name
    parent_file = FLAGS.parent_file
    file_type = FLAGS.file_type

    if file_type == 'arff':
        X_train, y_train, X_test, y_test = load_mv_ucr_data_arff(dataset_name, parent_file)
    elif file_type == 'csv':
        X_train, y_train, X_test, y_test = load_mv_ucr_data_csv(parent_file, dataset_name)
    else:
        raise ValueError("Unsupported file type. Use 'arff' or 'csv'.")

    # Save the dataset to pickle for later use
    output_file = os.path.join("Dataset", f"{dataset_name}.pkl")
    os.makedirs("Dataset", exist_ok=True)
    with open(output_file, 'wb') as f:
        pkl.dump([X_train, y_train, X_test, y_test], f)

    # Save dataset parameters
    with open('datasets_parameters.json', 'r') as jf:
        info = json.load(jf)
    info[dataset_name] = {
        "path": f"Dataset/{dataset_name}.pkl",
        "SEG_SIZE": X_train.shape[1],  # Adjusted to [1] since CSV has different shape
        "CHANNEL_NB": X_train.shape[2],
        "CLASS_NB": len(np.unique(y_train)) if len(np.unique(y_train)) > 1 else 1
    }
    with open('datasets_parameters.json', 'w') as jf:
        json.dump(info, jf, indent=2)

if __name__ == "__main__":
    flags.DEFINE_string('dataset_name', None, 'Dataset name')
    flags.DEFINE_string('parent_file', None, 'Parent directory containing the data files')
    flags.DEFINE_string('file_type', 'arff', 'File type: arff or csv')

    app.run(main)
