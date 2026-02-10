## Helsinki Tram MLOps Project

### Introduction
This project collects, processes, and analyzes real-time tram data from Helsinki using the HSL MQTT broker. The goal is to build a data pipeline for operational insights and machine learning applications, such as delay prediction and route optimization.

### Concept
1. **Data Ingestion:** Live tram telemetry is collected from the HSL MQTT broker and stored in a JSONL file.
2. **Data Processing:** The raw data is cleaned and transformed using Python scripts for further analysis.
3. **Analysis & Visualization:** Key statistics (e.g., average delays) are calculated and visualized in an interactive dashboard using Streamlit.
4. **Machine Learning (optional):** The processed data can be used to build predictive models for tram delays or other analytics.

