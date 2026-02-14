import os
import pandas as pd
from datetime import datetime, timedelta
from data_pipeline.data_fetcher import fetch_air_quality, fetch_weather
from data_pipeline.engineer_features import (
    aggregate_to_daily, 
    add_temporal_features, 
    add_statistical_features, 
    add_target_features, 
    clean_and_validate
)

# Configuration
DATA_DIR = "data/feature_store"
FILE_PATH = os.path.join(DATA_DIR, "karachi_daily_features.csv")

def run_pipeline(days_back=30):
    """
    Orchestrates the data pipeline for a given historical range.
    """
    print(f"--- Starting AQI Pipeline (Last {days_back} days) ---")
    
    # 1. Fetch
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Step 1: Fetching raw data from {start_date} to {end_date}...")
    # Fetching in bulk is usually faster for Open-Meteo
    aq_df = fetch_air_quality(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    w_df = fetch_weather(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    
    if aq_df is None or w_df is None:
        print("Pipeline aborted: Fetch failed.")
        return
    
    merged_hourly = pd.merge(aq_df, w_df, on="time", how="inner")
    
    # 2. Process & Engineer
    print("Step 2: Performing daily aggregation and feature engineering...")
    df = aggregate_to_daily(merged_hourly)
    df = add_temporal_features(df)
    df = add_statistical_features(df)
    df = add_target_features(df)
    df = clean_and_validate(df)
    
    # 3. Save to Local Feature Store (CSV)
    print(f"Step 3: Saving {len(df)} records to {FILE_PATH}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FILE_PATH, index=False)
    
    # 4. Upload to Hopsworks Cloud
    print("Step 4: Uploading to Hopsworks Cloud...")
    try:
        from data_pipeline.hopsworks_connector import create_or_get_feature_group
        # Ensure UTC and correct metadata for Hopsworks
        upload_df = df.copy()
        upload_df["event_timestamp"] = pd.to_datetime(upload_df["event_timestamp"])
        upload_df["karachi_id"] = "karachi_001"
        create_or_get_feature_group(upload_df)
        print("Success: Data synced with Hopsworks.")
    except Exception as e:
        print(f"Warning: Hopsworks upload failed: {e}")

    print("--- Pipeline Completed Successfully ---")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AQI Data Pipeline")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch back")
    args = parser.parse_args()
    
    run_pipeline(days_back=args.days)
