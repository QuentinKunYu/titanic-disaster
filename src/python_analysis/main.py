import pandas as pd
import os

print("--- Python Script Starting ---")

# Define the path to the data file
# This path assumes the script is run from the root of the project by Docker
data_path = "src/data/train.csv"

print(f"Attempting to load dataset from: {data_path}")

# Check if the file exists before trying to load it
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}")
    print("Please make sure 'train.csv' is in the 'src/data' folder.")
    # Print current working directory and its contents for debugging
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Contents of 'src/data': {os.listdir('src/data') if os.path.exists('src/data') else 'src/data directory not found'}")
else:
    # Load the training data
    try:
        train_df = pd.read_csv(data_path)
        print("Dataset loaded successfully!")

        # Print the shape and first 5 rows to confirm it's loaded correctly
        print("\nDataset Shape:", train_df.shape)
        print("\nFirst 5 rows of the dataset:")
        print(train_df.head())

    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")

print("\n--- Python Script Finished ---")