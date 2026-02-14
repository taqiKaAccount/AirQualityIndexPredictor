import pandas as pd
import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

load_dotenv()

from data_pipeline.hopsworks_connector import get_feature_store
from ml_pipeline.model_utils import load_latest_model_path

app = FastAPI(title="Pearls AQI Predictor API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to cache models and feature group
MODELS = {}
FEATURE_GROUP = None

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

@app.on_event("startup")
async def startup_event():
    """Initializes Hopsworks connections and loads available models."""
    global MODELS, FEATURE_GROUP
    try:
        print("Connecting to Hopsworks...")
        # 1. Load Feature Group handle (Version 4 with log features)
        fs = get_feature_store()
        FEATURE_GROUP = fs.get_feature_group("karachi_aqi_daily", version=5)
        print("Feature group handle (v4) retrieved.")

        # 2. Load Model Architectures
        # Map user-friendly names to registry names
        model_map = {
            "HGBR": "karachi_aqi_hgbr",
            "Random Forest": "karachi_aqi_rf",
            "Ridge Regression": "karachi_aqi_ridge",
            "Decision Tree": "karachi_aqi_dt",
            "Deep Learning (MLP)": "karachi_aqi_mlp",
            "HGBR (Optimized)": "karachi_aqi_hgbr", # Backwards compatibility
            "Optimized": "karachi_aqi_hgbr"
        }

        print("Loading model architectures from Registry...")
        for display_name, registry_name in model_map.items():
            try:
                model_path = load_latest_model_path(registry_name)
                if model_path:
                    MODELS[display_name] = joblib.load(model_path)
                    print(f"Loaded {display_name} ({registry_name})")
            except Exception as e:
                print(f"Warning: Could not load {display_name}: {e}")

    except Exception as e:
        print(f"Error during startup: {e}")

def get_latest_data():
    """Fetches the latest row from Hopsworks Feature Group."""
    if FEATURE_GROUP:
        df = FEATURE_GROUP.read()
        df = df.sort_values("event_timestamp").reset_index(drop=True)
        return df
    return None

@app.get("/")
async def root():
    return {"message": "Welcome to Pearls AQI Predictor API", "city": "Karachi", "status": "Multi-Model Support Active"}

@app.get("/current")
async def get_current_aqi():
    df = get_latest_data()
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        return {
            "timestamp": str(latest["event_timestamp"]),
            "aqi": round(float(latest["aqi"]), 2),
            "category": get_aqi_category(latest["aqi"]),
            "temperature": round(float(latest["temperature"]), 1),
            "humidity": round(float(latest["humidity"]), 1),
            "wind_speed": round(float(latest["windspeed"]), 1)
        }
    
    return {"error": "No data available from Feature Store"}

@app.get("/predict")
async def get_predictions(model_name: str = "HGBR"):
    df = get_latest_data()
    
    # Check if requested model exists
    selected_model = MODELS.get(model_name)
    if selected_model is None:
        # Fallback to HGBR if available
        selected_model = MODELS.get("HGBR")
    
    if df is None or df.empty or selected_model is None:
        return {"error": f"Model '{model_name}' or features not available"}

    latest = df.iloc[-1]
    feature_cols = selected_model["features"]
    X_input = df[feature_cols].iloc[-1:]
    
    # Get overall 1d performance (representative R2)
    # The 'performance' dict was added in the recent training run
    r2_score = selected_model.get("performance", {}).get("1d", {}).get("R2", 0.0)
    
    predictions = []
    base_date = pd.to_datetime(latest["event_timestamp"])
    
    # Run Inference per Horizon
    for i, horizon in enumerate(["1d", "2d", "3d"]):
        h_meta = selected_model["models"][horizon]
        m = h_meta["model"]
        s = h_meta["scaler"]
        
        # Scale and predict
        X_scaled = s.transform(X_input)
        aqi_val = m.predict(X_scaled)[0]
        
        predictions.append({
            "date": (base_date + timedelta(days=i+1)).strftime("%Y-%m-%d"),
            "aqi": round(float(aqi_val), 2),
            "category": get_aqi_category(aqi_val)
        })
    
    return {
        "model": model_name,
        "r2_score": round(float(r2_score), 3),
        "forecast": predictions
    }

@app.get("/history")
async def get_history(days: int = 30):
    df = get_latest_data()
    if df is not None and not df.empty:
        history_df = df.tail(days)
        history = []
        for _, row in history_df.iterrows():
            history.append({
                "date": str(row["event_timestamp"]),
                "aqi": round(float(row["aqi"]), 2)
            })
        return history
    
    return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
