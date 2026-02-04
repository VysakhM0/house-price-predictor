import pandas as pd
import os

# 1. CONSTANTS
# We read directly from the 'data' folder now
RAW_DATA_PATH = os.path.join("data", "house_prices_dataset.csv")
CLEANED_DATA_PATH = os.path.join("data", "housing_cleaned.csv")

def load_data(filepath):
    """
    Loads data from a local CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}. Did you move the CSV to the 'data' folder?")
    
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Performs basic cleaning.
    Since this specific dataset is clean, we just remove duplicates.
    """
    print("Cleaning data...")
    
    # 1. Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    
    if initial_count > final_count:
        print(f"Removed {initial_count - final_count} duplicate rows.")
    else:
        print("No duplicates found.")

    # 2. (Optional) You could rename columns here if they had spaces
    # df = df.rename(columns={'distance_to_city(km)': 'distance_km'})
    
    return df

def save_data(df, filepath):
    """
    Saves the cleaned dataframe.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Cleaned data saved to {filepath}")

if __name__ == "__main__":
    # A. Load
    df = load_data(RAW_DATA_PATH)
    
    # B. Clean
    df_clean = clean_data(df)
    
    # C. Save
    save_data(df_clean, CLEANED_DATA_PATH)
    
    # D. Quick check
    print("\nData Preview:")
    print(df_clean.head())