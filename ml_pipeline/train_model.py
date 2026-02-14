import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from dotenv import load_dotenv

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV

from data_pipeline.hopsworks_connector import get_feature_store

load_dotenv()

def fetch_data_from_hopsworks(fg_name="karachi_aqi_daily", version=5):
    """
    Fetches the data from Hopsworks Feature Store.
    """
    fs = get_feature_store()
    fg = fs.get_feature_group(name=fg_name, version=version)
    
    # Read the data
    df = fg.read()
    
    # Sort by date for safety
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    return df

def get_model_constructor(model_type):
    """
    Returns the constructor and default params for the requested architecture.
    """
    if model_type == "hgbr":
        return HistGradientBoostingRegressor(
            max_iter=200,           
            max_depth=3,             # reduce depth to avoid overfitting
            learning_rate=0.05,       # lower LR often helps with small data
            l2_regularization=0.1,    # add L2 regularization
            early_stopping=True,      # use validation to stop
            validation_fraction=0.1,  # 10% of data for validation
            n_iter_no_change=10,      # patience
            random_state=42
            )
    elif model_type == "rf":
        return RandomForestRegressor(
            n_estimators=150,          
            max_depth=8,                
            min_samples_split=10,       
            min_samples_leaf=5,         
            bootstrap=True,
            random_state=42
            )
    elif model_type == "ridge":
        return Ridge(alpha=1.0)
    elif model_type == "dt":
        return DecisionTreeRegressor(
            max_depth=5,                
            min_samples_split=15,
            min_samples_leaf=10,
            max_features='sqrt',         
            ccp_alpha=0.01,              
            random_state=42
            )
    elif model_type == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),          # No trailing 1 – two hidden layers
            activation='relu',
            alpha=0.001,                           # Lower regularization than 0.01
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,                          # Give it more iterations
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_and_evaluate_all():
    """
    Trains and registers multiple model architectures for comparison.
    """
    print("--- Starting Multi-Model Training Pipeline ---")
    
    # 1. Fetch Data
    print("Step 1: Fetching data from Hopsworks (v5)...")
    df = fetch_data_from_hopsworks()
    
    # 2. Prepare Data Structure
    target_cols = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]
    drop_cols = ["event_timestamp", "karachi_id", "date", "created"]
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in target_cols]
    
    model_types = ["hgbr", "rf", "ridge", "dt", "mlp"]
    horizons = ["1d", "2d", "3d"]
    
    all_performance = {}

    for m_type in model_types:
        print(f"\n--- Training {m_type.upper()} Architecture ---")
        performance = {}
        horizon_models = {}
        
        for horizon in horizons:
            target = f"target_aqi_{horizon}"
            print(f"Training {m_type} for {horizon} horizon...")
            
            df_target = df.dropna(subset=[target]).copy()
            X = df_target[feature_cols]
            y = df_target[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = get_model_constructor(m_type)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            y_persistence = X_test["aqi"]
            r2_persistence = r2_score(y_test, y_persistence)
            
            performance[horizon] = {
                "RMSE": rmse, 
                "MAE": mae, 
                "R2": r2, 
                "R2_Persistence": r2_persistence
            }
            horizon_models[horizon] = {"model": model, "scaler": scaler}
            print(f"{horizon} {m_type} -> R2: {r2:.3f} (Persistence R2: {r2_persistence:.3f})")

        # 4. Save and Register this architecture set
        registry_name = f"karachi_aqi_{m_type}"
        local_path = f"models/aqi_{m_type}_models.pkl"
        os.makedirs("models", exist_ok=True)
        
        joblib.dump({
            "models": horizon_models, 
            "features": feature_cols,
            "performance": performance
        }, local_path)
        print(f"Saved {m_type} models to {local_path}")
        
        print(f"Uploading {m_type} to Model Registry as '{registry_name}'...")
        try:
            from ml_pipeline.model_utils import save_model_to_registry
            save_model_to_registry(
                model_path=local_path, 
                model_name=registry_name,
                metrics=performance["1d"],
                description=f"{m_type.upper()} models for 1-3 day AQI forecasts. Multi-model suite."
            )
        except Exception as e:
            print(f"Warning: Model registration failed for {m_type}: {e}")
            
        all_performance[m_type] = performance

    return all_performance

if __name__ == "__main__":
    results = train_and_evaluate_all()
    print("\nFinal Performance Summary (1-Day R2):")
    for m_type, perfs in results.items():
        print(f"{m_type.upper():<10}: {perfs['1d']['R2']:.3f} (Persist: {perfs['1d']['R2_Persistence']:.3f})")
