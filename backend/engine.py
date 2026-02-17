"""
AQIEngine: Core business logic for the AQI Predictor.

This module encapsulates all Hopsworks connections, model loading,
and prediction logic. It can be used by:
  - FastAPI backend (backend/main.py) for API-based deployments
  - Streamlit frontend (frontend/app.py) for Streamlit Cloud deployments
"""

import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Also load Streamlit secrets if running on Streamlit Cloud
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        # Inject Streamlit secrets into environment variables
        for key in ["HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT"]:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
except ImportError:
    pass


def get_aqi_category(aqi):
    """Returns the human-readable AQI category."""
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


class AQIEngine:
    """
    Encapsulates all AQI prediction logic, including Hopsworks
    connectivity, model loading, and inference.
    """

    def __init__(self):
        self.models = {}
        self.feature_group = None
        self._initialized = False

    def startup(self):
        """Initializes Hopsworks connections and loads available models."""
        if self._initialized:
            return

        from data_pipeline.hopsworks_connector import get_feature_store
        from ml_pipeline.model_utils import load_latest_model_path

        try:
            print("Connecting to Hopsworks...")
            fs = get_feature_store()
            self.feature_group = fs.get_feature_group("karachi_aqi_daily", version=5)
            print("Feature group handle (v5) retrieved.")

            model_map = {
                "HGBR": "karachi_aqi_hgbr",
                "Random Forest": "karachi_aqi_rf",
                "Ridge Regression": "karachi_aqi_ridge",
                "Decision Tree": "karachi_aqi_dt",
                "Deep Learning (MLP)": "karachi_aqi_mlp",
                "HGBR (Optimized)": "karachi_aqi_hgbr",
                "Optimized": "karachi_aqi_hgbr",
            }

            print("Loading model architectures from Registry...")
            for display_name, registry_name in model_map.items():
                try:
                    model_path = load_latest_model_path(registry_name)
                    if model_path:
                        self.models[display_name] = joblib.load(model_path)
                        print(f"Loaded {display_name} ({registry_name})")
                except Exception as e:
                    print(f"Warning: Could not load {display_name}: {e}")

            self._initialized = True
        except Exception as e:
            print(f"Error during startup: {e}")
            raise

    def _get_latest_data(self):
        """Fetches the latest rows from Hopsworks Feature Group."""
        if self.feature_group:
            df = self.feature_group.read()
            df = df.sort_values("event_timestamp").reset_index(drop=True)
            return df
        return None

    def get_current_aqi(self):
        """Returns the current AQI data dict."""
        df = self._get_latest_data()
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            return {
                "timestamp": str(latest["event_timestamp"]),
                "aqi": round(float(latest["aqi"]), 2),
                "category": get_aqi_category(latest["aqi"]),
                "temperature": round(float(latest["temperature"]), 1),
                "humidity": round(float(latest["humidity"]), 1),
                "wind_speed": round(float(latest["windspeed"]), 1),
            }
        return {"error": "No data available from Feature Store"}

    def get_predictions(self, model_name="HGBR"):
        """Returns forecast predictions for the next 3 days."""
        df = self._get_latest_data()

        selected_model = self.models.get(model_name)
        if selected_model is None:
            selected_model = self.models.get("HGBR")

        if df is None or df.empty or selected_model is None:
            return {"error": f"Model '{model_name}' or features not available"}

        latest = df.iloc[-1]
        feature_cols = selected_model["features"]
        X_input = df[feature_cols].iloc[-1:]

        performance = selected_model.get("performance", {})

        predictions = []
        base_date = pd.to_datetime(latest["event_timestamp"])

        for i, horizon in enumerate(["1d", "2d", "3d"]):
            h_meta = selected_model["models"][horizon]
            m = h_meta["model"]
            s = h_meta["scaler"]

            X_scaled = s.transform(X_input)
            aqi_val = m.predict(X_scaled)[0]

            # Per-horizon metrics from training
            h_perf = performance.get(horizon, {})

            predictions.append({
                "date": (base_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "aqi": round(float(aqi_val), 2),
                "category": get_aqi_category(aqi_val),
                "r2": round(float(h_perf.get("R2", 0.0)), 3),
                "mae": round(float(h_perf.get("MAE", 0.0)), 2),
                "rmse": round(float(h_perf.get("RMSE", 0.0)), 2),
            })

        return {
            "model": model_name,
            "forecast": predictions,
        }

    def get_history(self, days=30):
        """Returns historical AQI data."""
        df = self._get_latest_data()
        if df is not None and not df.empty:
            history_df = df.tail(days)
            history = []
            for _, row in history_df.iterrows():
                history.append({
                    "date": str(row["event_timestamp"]),
                    "aqi": round(float(row["aqi"]), 2),
                })
            return history
        return []
