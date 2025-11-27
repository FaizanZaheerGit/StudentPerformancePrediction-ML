import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded from: {file_path} (shape={df.shape})")
        return df
    except FileNotFoundError:
        print(f"Error: file not found -> {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error: file is not a valid CSV -> {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error reading data: {e}")
        return None


def save_data(df, filename="processed_data.csv"):

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, filename)
    try:
        df.to_csv(save_path, index=False)
        print(f"Processed data saved to: {save_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")
