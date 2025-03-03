from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Optional
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our transformations with the CORRECT function name
from scripts.transformations import pace_to_minutes, minutes_to_pace_str

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pace_prediction_api")

# Initialize FastAPI
app = FastAPI(title="Running Pace Prediction API")

# Define the request model
class RunData(BaseModel):
    distance: float
    elapsed_time: Optional[float] = None
    average_heart_rate: Optional[float] = None
    average_cadence: Optional[float] = None
    total_elevation_gain: Optional[float] = None
    average_grade: Optional[float] = None
    relative_effort: Optional[float] = None
    # Include any optional historical data
    last_7_runs: Optional[List[Dict]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "distance": 8.05,  # km
                "elapsed_time": 3600,  # seconds
                "average_heart_rate": 155,
                "average_cadence": 170,
                "total_elevation_gain": 120,
                "average_grade": 2.5,
                "relative_effort": 110,
                "last_7_runs": [
                    {
                        "distance": 5.2,
                        "elapsed_time": 1800,
                        "average_heart_rate": 150
                    }
                    # Add more runs here if needed
                ]
            }
        }

# Model loading
def load_latest_model():
    """Load the latest XGBoost model"""
    models_dir = "models"
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        model_folders = [os.path.join(models_dir, d) for d in os.listdir(models_dir) 
                        if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("xgboost_pace_")]
        
        if not model_folders:
            logger.warning("No model folders found. API will start but predictions won't work until a model is trained.")
            return None, None
        
        # Get the latest model based on timestamp in folder name
        latest_model_dir = max(model_folders)
        
        # Load model
        model_path = os.path.join(latest_model_dir, "model.xgb")
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Load feature names
        with open(os.path.join(latest_model_dir, "feature_names.json"), 'r') as f:
            feature_names = json.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        return model, feature_names
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.warning("API will start but predictions won't work until a model is trained.")
        return None, None

# Load the model and feature names at startup
model, feature_names = load_latest_model()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Running Pace Prediction API"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_status": "ready" if model is not None else "not_loaded"
    }

def prepare_input_features(run_data: RunData) -> pd.DataFrame:
    """Prepare input features from the request data"""
    # Start with the direct features
    data = {
        "distance": run_data.distance,
        "elapsed_time": run_data.elapsed_time,
        "average_heart_rate": run_data.average_heart_rate,
        "average_cadence": run_data.average_cadence,
        "total_elevation_gain": run_data.total_elevation_gain,
        "average_grade": run_data.average_grade,
        "relative_effort": run_data.relative_effort
    }
    
    # Process and add historical data if available
    if run_data.last_7_runs:
        # Calculate rolling metrics from historical data
        distances = [run.get("distance", 0) for run in run_data.last_7_runs if "distance" in run]
        hrs = [run.get("average_heart_rate", 0) for run in run_data.last_7_runs if "average_heart_rate" in run]
        
        # Add rolling metrics if we have data
        if distances:
            data["7d_avg_distance"] = sum(distances) / len(distances)
            data["7d_total_distance"] = sum(distances)
        
        if hrs:
            data["7d_avg_heart_rate"] = sum(hrs) / len(hrs)
    
    # Create DataFrame with just the features needed by the model
    df = pd.DataFrame([data])
    
    # Filter to include only the features used by the model
    common_features = [col for col in df.columns if col in feature_names]
    missing_features = [col for col in feature_names if col not in df.columns]
    
    # Fill missing features with zeros
    for col in missing_features:
        df[col] = 0
    
    # Ensure all model features are present
    df = df[feature_names]
    
    return df

@app.post("/predict")
def predict_pace(run_data: RunData):
    """Predict running pace for a given set of features"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="No trained model is available. Please train a model first using train_xgboost.py."
        )
        
    try:
        # Prepare input features
        input_df = prepare_input_features(run_data)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(input_df)
        
        # Make prediction
        pace_minutes = model.predict(dmatrix)[0]
        
        # Convert to pace string - using the correct function name
        pace_str = minutes_to_pace_str(pace_minutes)
        
        return {
            "predicted_pace_minutes": pace_minutes,
            "predicted_pace": pace_str,
            "inputs": {
                "distance": run_data.distance,
                "features_used": len(feature_names) if feature_names else 0
            }
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# For testing locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 