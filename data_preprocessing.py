# Module for data loading and preprocessing

def load_data(file_path):
    """
    Loads data from a specified file path
    """
    data = pd.read_csv(file_path)
    return data

def inspect_data(data):
    """
    Inspect the basic structure of the dataset
    """
    print(f"Shape of data: {data.shape}")
    print("Top 5 rows of data: ")
    print(data.head())

# Additional preprocessing steps can be added here
