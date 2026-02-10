import pandas as pd
import json
import time 

def load_data(file_path):
    """Load tram data from a JSONL file into a pandas DataFrame."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    return pd.DataFrame(records)

    

def clean_data(df):
    """fix timestamps and remove junk columns."""
    #convert tst to actual time object
    df['timestamp'] = pd.to_datetime(df['tst'], unit='s')
    # add an hour column for easier analysis
    df['hour'] = df['timestamp'].dt.hour
    # drop rows where delay (dl) is missing
    return df.dropna(subset=['dl'])

def run_analysis(df):
    """Step 3: Calculate the stats"""
    print(f"Overall Avg Delay: {df['dl'].mean():.2f}s")
    print("\nAvg Delay per Hour:")
    print(df.groupby('hour')['dl'].mean())



if __name__ == "__main__":
    raw_df = load_data("../raw_tram_data.jsonl")
    clean_df = clean_data(raw_df)
    run_analysis(clean_df)