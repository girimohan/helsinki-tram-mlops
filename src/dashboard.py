import pandas as pd
import json
import streamlit as st

# ...existing code...
def load_data(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)

# ...existing code...
def clean_data(df):
    df['timestamp'] = pd.to_datetime(df['tst'], unit='s')
    df['hour'] = df['timestamp'].dt.hour
    return df.dropna(subset=['dl'])

# Load and clean data
df = load_data("../raw_tram_data.jsonl")
df = clean_data(df)

st.title("Helsinki Tram Delay Dashboard")

st.write(f"**Overall Avg Delay:** {df['dl'].mean():.2f} seconds")

st.subheader("Average Delay per Hour")
hourly_delay = df.groupby('hour')['dl'].mean()
st.bar_chart(hourly_delay)

st.subheader("Average Delay per Tram")
tram_delay = df.groupby('veh')['dl'].mean().sort_values(ascending=False)
st.dataframe(tram_delay)
