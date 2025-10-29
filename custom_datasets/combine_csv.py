import os
import pandas as pd
import random

def process_csv_files(directory):
    """
    Reads all CSV files in the given directory, extracts the last column
    from each, combines them into a single DataFrame (one column per file),
    adds a random integer as the target column, and saves it as a single CSV
    named after the parent directory.

    :param directory: Path to the directory containing the CSV files.
    """
    # Get the directory name
    dir_name = os.path.basename(os.path.normpath(directory))
    output_file = os.path.join(directory, f"{dir_name}.csv")

    print(f"Processing directory: {directory}")
    print(f"Output file: {output_file}")

    combined_data = {}  # To store the last column of each CSV

    # Iterate over all files in the directory
    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith('.csv') and file_name != f"{dir_name}.csv":  # Process only CSV files
            file_path = os.path.join(directory, file_name)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the file has at least one column
                if df.empty or df.shape[1] < 1:
                    print(f"  Skipping empty or invalid file: {file_name}")
                    continue
                
                # Extract the last column and add it to the combined data
                column_name = os.path.splitext(file_name)[0]  # Use file name as column name
                combined_data[column_name] = df.iloc[:, -1].values

                print(f"  Processed {file_name} ({len(df)} rows)")

            except Exception as e:
                print(f"  Error processing file {file_name}: {e}")

    # Create a DataFrame from the combined data
    if combined_data:
        # Find the minimum length across all columns
        min_length = min(len(values) for values in combined_data.values())
        max_length = max(len(values) for values in combined_data.values())
        
        if min_length != max_length:
            print(f"\n⚠ Warning: CSV files have different lengths (min: {min_length}, max: {max_length})")
            print(f"  Truncating all columns to {min_length} rows to match shortest file")
            
            # Truncate all arrays to the minimum length
            combined_data = {key: values[:min_length] for key, values in combined_data.items()}
        
        combined_df = pd.DataFrame(combined_data)
        random_target = random.randint(0, 10)
        random_target = 98
        # Add a random integer column as the target
        combined_df['Target'] = [random_target for _ in range(len(combined_df))]

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Combined CSV saved as: {output_file}")
        print(f"  Shape: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
    else:
        print("\n✗ No valid CSV files were found or processed.")

# Example usage
if __name__ == "__main__":
    # Set the subdirectory name here
    subdir = "nginx"  # Change this to the desired subdirectory
    target_dir = os.path.join("custom_datasets", subdir)
    
    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        print("\nAvailable subdirectories in custom_datasets/:")
        
        custom_datasets_dir = "custom_datasets"
        if os.path.isdir(custom_datasets_dir):
            subdirs = [d for d in os.listdir(custom_datasets_dir) 
                      if os.path.isdir(os.path.join(custom_datasets_dir, d))]
            for subdir_name in sorted(subdirs):
                print(f"  - {subdir_name}")
    else:
        process_csv_files(target_dir)
