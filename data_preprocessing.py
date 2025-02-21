import pandas as pd
import argparse

def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', header=0)
    df.columns = df.columns.str.strip()  
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    return df

def compute_price_change(df):
    df['Close_Change'] = df['Close'].pct_change(periods=15) * 100
    return df

def categorize_price_change(df):
    bins = [-float("inf"), -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")]
    labels = [-11,-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  

    df['Label'] = pd.cut(df['Close_Change'], bins=bins, labels=labels)
    df['Label'] = df['Label'].fillna(0).astype(int)
    df['Close_Change'] = df['Close_Change'].fillna(0).astype(int)
    return df

def normalize_data(df):
    cols_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[cols_to_normalize] = (df[cols_to_normalize] - df[cols_to_normalize].mean()) / df[cols_to_normalize].std()
    return df

def preprocess_data(input_file, output_file):
    df = load_data(input_file)
    df = compute_price_change(df)
    df = categorize_price_change(df)
    df = normalize_data(df)
    df.to_csv(output_file, index=True)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OHLC data")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed CSV file")
    
    args = parser.parse_args()
    preprocess_data(args.input, args.output)
