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
    # Get the parent directory name
    parent_dir_name = os.path.basename(os.path.normpath(directory))
    output_file = f"{directory}/{parent_dir_name}.csv"

    print(output_file)

    combined_data = {}  # To store the last column of each CSV

    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv') and file_name != f"{parent_dir_name}.csv":  # Process only CSV files
            file_path = os.path.join(directory, file_name)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the file has at least one column
                if df.empty or df.shape[1] < 1:
                    print(f"Skipping empty or invalid file: {file_name}")
                    continue
                
                # Extract the last column and add it to the combined data
                column_name = os.path.splitext(file_name)[0]  # Use file name as column name
                combined_data[column_name] = df.iloc[:, -1].values

                print(f"Processed {file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Create a DataFrame from the combined data
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        random_target = random.randint(0, 10)
        # Add a random integer column as the target
        combined_df['Target'] = [ random_target for _ in range(len(combined_df))]

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(output_file, index=False)
        print(f"Combined CSV saved as: {output_file}")
    else:
        print("No valid CSV files were found or processed.")

# Example usage
if __name__ == "__main__":
    parent_dir = "./custom_datasets/"
    print(os.listdir(parent_dir))

    for dir_path in os.listdir("./custom_datasets/"):
        dir_path = parent_dir + dir_path
        if os.path.isdir(dir_path):
            process_csv_files(dir_path)
