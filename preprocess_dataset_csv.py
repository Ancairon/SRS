import os
import glob
import numpy as np
import pandas as pd
import json  # Add json import
from sklearn.model_selection import train_test_split
import pickle as pkl
from scipy.io import arff


def load_mv_ucr_data_csv(parent_file, dataset_name):
    # Extract Data Dimensions
    dim_df = pd.read_csv("DataDimensions.csv")
    ds_idx = dim_df[dim_df["Problem"] == dataset_name].index[0]
    ds_trn_size = int(dim_df.at[ds_idx, "TrainSize"])
    ds_channel_nb = int(dim_df.at[ds_idx, "NumDimensions"])
    ds_seg_size = int(dim_df.at[ds_idx, "SeriesLength"])
    ds_test_size = int(dim_df.at[ds_idx, "TestSize"])

    # Find all CSV files with the dataset name as prefix
    csv_files = sorted(glob.glob(os.path.join(parent_file, f"{dataset_name}*.csv")))
    csv_files = csv_files[:-1]

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ds_test_size, random_state=42)


    return X_train, y_train, X_test, y_test

def main(dataset,parent_dir):
    # dataset = FLAGS.dataset_name
    # parent_dir = FLAGS.parent_file

    X_train, y_train, X_test, y_test = load_mv_ucr_data_csv(parent_dir, dataset)

    # Save the dataset to pickle for later use
    output_file = os.path.join("Dataset", f"{dataset}.pkl")
    os.makedirs("Dataset", exist_ok=True)
    with open(output_file, 'wb') as f:
        pkl.dump([X_train, y_train, X_test, y_test], f)

    # Save dataset parameters
    with open('datasets_parameters.json', 'r') as jf:
        info = json.load(jf)
    info[dataset] = {
        "path": f"Dataset/{dataset}.pkl",
        "SEG_SIZE": X_train.shape[1],  # Adjusted to [1] since CSV has different shape
        "CHANNEL_NB": X_train.shape[2],
        "CLASS_NB": len(np.unique(y_train)) if len(np.unique(y_train)) > 1 else 1
    }
    with open('datasets_parameters.json', 'w') as jf:
        json.dump(info, jf, indent=2)

if __name__ == "__main__":
    
    parent_dir = 'custom_datasets'  # Parent directory containing subdirectories for each dataset

    datasets = [os.path.join(parent_dir, d, f"{d}.csv") 
                for d in os.listdir(parent_dir) 
                if os.path.isdir(os.path.join(parent_dir, d))]
    for file in datasets:
        dataset = file.split("\\")[2].replace(".csv","")
        parent_dir="./custom_datasets/"+dataset
        print(dataset, parent_dir)


        main(dataset, parent_dir)
