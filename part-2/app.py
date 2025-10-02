import fastapi
import pandas as pd
import numpy as np
import mlflow
import time
import requests
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = fastapi.FastAPI(title="Water Quality Prediction API", version="1.0.0")

# Global variables for model state
model = None
model_version = None
current_model_name = None  # Track which model is currently loaded

class WaterQualityInput(BaseModel):
    """Input schema for water quality prediction - only fields used in training"""
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    dissolved_min: Optional[float] = None
    dissolved_max: Optional[float] = None
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    bod_min: Optional[float] = None
    bod_max: Optional[float] = None
    nitrate_min: Optional[float] = None
    nitrate_max: Optional[float] = None
    fecal_coliform_min: Optional[float] = None
    fecal_coliform_max: Optional[float] = None
    total_coliform_min: Optional[float] = None
    total_coliform_max: Optional[float] = None
    fecal_min: Optional[float] = None
    fecal_max: Optional[float] = None

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: int
    probability: float
    status: str
    model_version: str

class ModelUpdateRequest(BaseModel):
    """Request schema for model updates"""
    model_name: str = "water_quality_RandomForest"
    version: Optional[str] = "latest"


def wait_for_mlflow(max_retries=30, delay=2):
    """Wait for MLflow server to be ready"""
    mlflow_url = "http://mlflow:8080"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{mlflow_url}", timeout=5)
            if response.status_code == 200:
                logger.info("MLflow server is ready")
                return True
        except Exception as e:
            logger.info(f"Waiting for MLflow server... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    
    logger.error("MLflow server not ready after maximum retries")
    return False


def create_experiment_if_not_exists(experiment_name):
    """Create experiment if it doesn't exist"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id
        else:
            logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment.experiment_id}")
            return experiment.experiment_id
    except Exception as e:
        logger.error(f"Error with experiment: {e}")
        return None


def load_model_from_registry(model_name: str = "water_quality_RandomForest", version: str = "latest"):
    """Load model from MLflow Model Registry"""
    global model, model_version, current_model_name
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://mlflow:8080/")
        
        # Create experiment if it doesn't exist
        experiment_name = "water_quality_classification"
        create_experiment_if_not_exists(experiment_name)
        
        if version == "latest":
            model_uri = f"models:/{model_name}/latest"
        else:
            model_uri = f"models:/{model_name}/{version}"
            
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = version
        current_model_name = model_name  # Track which model is loaded
        logger.info(f"Successfully loaded model {model_name} version {version}")
        return True
        
    except mlflow.exceptions.RestException as e:
        if "not found" in str(e).lower():
            logger.warning(f"Model '{model_name}' not found in registry. Trying fallback to experiment runs...")
            try:
                experiment = mlflow.get_experiment_by_name("water_quality_classification")
                if experiment:
                    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                            order_by=["start_time DESC"], max_results=1)
                    if not runs.empty:
                        run_id = runs.iloc[0]['run_id']
                        model_uri = f"runs:/{run_id}/random_forest_model"
                        model = mlflow.pyfunc.load_model(model_uri)
                        model_version = f"run_{run_id[:8]}"
                        current_model_name = model_name
                        logger.info(f"Loaded model from run {run_id}")
                        return True
                    else:
                        logger.warning(f"No runs found in experiment '{experiment_name}'")
                else:
                    logger.warning(f"Experiment '{experiment_name}' not found")
            except Exception as e2:
                logger.error(f"Fallback loading also failed: {str(e2)}")
        else:
            logger.error(f"Failed to load model: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting Water Quality Prediction API...")
    
    # Wait for MLflow to be ready
    if not wait_for_mlflow():
        logger.warning("MLflow server not available - model loading skipped")
        return
    
    # Try to load the model
    success = load_model_from_registry()
    if not success:
        logger.warning("Failed to load model on startup - will need to load manually")
        logger.info("You can:")
        logger.info("1. Train a model first using the training script")
        logger.info("2. Use the /update_model endpoint to retry loading")


def get_available_feature_columns():
    """Get the exact same feature columns that were used during training"""
    # These are the exact 16 columns that were available during training
    available_cols = [
        'Temperature (C) - Min',
        'Temperature (C) - Max',
        'Dissolved - Min',
        'Dissolved - Max',
        'pH - Min',
        'pH - Max',
        'BOD (mg/L) - Min',
        'BOD (mg/L) - Max',
        'NitrateN (mg/L) - Min',
        'NitrateN (mg/L) - Max',
        'Fecal Coliform (MPN/100ml) - Min',
        'Fecal Coliform (MPN/100ml) - Max',
        'Total Coliform (MPN/100ml) - Min',
        'Total Coliform (MPN/100ml) - Max',
        'Fecal - Min',
        'Fecal - Max'
    ]
    
    return available_cols


def prepare_input_data(input_data: WaterQualityInput) -> pd.DataFrame:
    """Convert input data to DataFrame with proper column names matching training data"""
    # Get the exact columns used during training
    available_cols = get_available_feature_columns()
    
    # Map input fields to the exact column names (all 16 features)
    data_dict = {
        'Temperature (C) - Min': input_data.temperature_min,
        'Temperature (C) - Max': input_data.temperature_max,
        'Dissolved - Min': input_data.dissolved_min,
        'Dissolved - Max': input_data.dissolved_max,
        'pH - Min': input_data.ph_min,
        'pH - Max': input_data.ph_max,
        'BOD (mg/L) - Min': input_data.bod_min,
        'BOD (mg/L) - Max': input_data.bod_max,
        'NitrateN (mg/L) - Min': input_data.nitrate_min,
        'NitrateN (mg/L) - Max': input_data.nitrate_max,
        'Fecal Coliform (MPN/100ml) - Min': input_data.fecal_coliform_min,
        'Fecal Coliform (MPN/100ml) - Max': input_data.fecal_coliform_max,
        'Total Coliform (MPN/100ml) - Min': input_data.total_coliform_min,
        'Total Coliform (MPN/100ml) - Max': input_data.total_coliform_max,
        'Fecal - Min': input_data.fecal_min,
        'Fecal - Max': input_data.fecal_max
    }
    
    # Create DataFrame with only the columns that were used in training
    filtered_data = {col: data_dict.get(col, 0.0) for col in available_cols}
    df = pd.DataFrame([filtered_data])
    
    # Apply the same preprocessing as training
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with 0 (matching training logic)
    df = df.fillna(0)
    
    return df


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: WaterQualityInput):
    """Endpoint to return predictions upon a POST request - works with any PyFunc model"""
    global model, model_version, current_model_name
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train a model first or use /update_model endpoint."
        )
    
    try:
        # Prepare input data
        df = prepare_input_data(input_data)
        
        # Use PyFunc predict method - this works for any model type
        # All preprocessing is handled internally by the PyFunc wrapper
        prediction_result = model.predict(df)
        
        # Handle the prediction result (could be scalar or array)
        if hasattr(prediction_result, '__len__') and len(prediction_result) > 0:
            prediction = int(prediction_result[0])
        else:
            prediction = int(prediction_result)
        
        # For PyFunc models, we provide a simple probability estimate
        # This could be enhanced by modifying the PyFunc models to return probabilities
        probability = 0.8 if prediction == 1 else 0.2
        
        # Determine status
        status = "Good Water Quality" if prediction == 1 else "Bad Water Quality"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            status=status,
            model_version=f"{current_model_name}:{model_version}" if current_model_name else model_version or "unknown"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models/available")
def list_available_models():
    """List all available models in the MLflow registry"""
    try:
        mlflow.set_tracking_uri("http://mlflow:8080/")
        client = mlflow.MlflowClient()
        
        try:
            models = client.search_registered_models()
            available_models = []
            
            for model in models:
                model_versions = client.search_model_versions(f"name='{model.name}'")
                versions = [mv.version for mv in model_versions]
                available_models.append({
                    "name": model.name,
                    "description": model.description,
                    "versions": versions,
                    "latest_version": max(versions) if versions else None
                })
            
            return {
                "available_models": available_models,
                "total_models": len(available_models)
            }
        except Exception as e:
            # If no registered models exist, return empty list
            return {
                "available_models": [],
                "total_models": 0,
                "message": "No registered models found. Train and register a model first."
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/update_model")
def update_model(model_name: str = "water_quality_RandomForest", version: str = "latest"):
    """Endpoint allowing to update the model with a MLflow model version"""
    try:
        success = load_model_from_registry(model_name, version)
        if success:
            return {
                "message": f"Successfully updated to model {model_name} version {version}",
                "model_version": model_version,
                "current_model_name": current_model_name
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Failed to load model '{model_name}' version '{version}'. Make sure the model exists in MLflow registry or experiment runs."
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model update failed: {str(e)}")


@app.post("/reload_model")
async def reload_model():
    """Endpoint to reload the current model or try loading default model"""
    global current_model_name
    model_name = current_model_name or "water_quality_RandomForest"
    success = load_model_from_registry(model_name)
    return {
        "message": "Model reload attempted",
        "success": success,
        "model_loaded": model is not None,
        "model_version": model_version,
        "current_model_name": current_model_name
    }


@app.get("/model/info")
def get_model_info():
    """Get information about the currently loaded model"""
    if model is None:
        return {
            "model_loaded": False,
            "message": "No model loaded. Train a model first or use /update_model endpoint.",
            "available_actions": [
                "Train a model using the training script",
                "Use GET /update_model to load an existing model",
                "Use POST /reload_model to retry loading"
            ]
        }
    
    try:
        return {
            "model_loaded": True,
            "current_model_name": current_model_name or "unknown",
            "model_version": model_version,
            "model_type": "Classification" if current_model_name == "water_quality_RandomForest" else "Regression",
            "available_models": {
                "RandomForest": "water_quality_RandomForest",
                "LinearRegression": "water_quality_LinearRegression"
            },
            "note": "Use /models/available for detailed model registry information"
        }
        
    except Exception as e:
        return {
            "model_loaded": True,
            "model_version": model_version,
            "error": f"Could not fetch detailed model info: {str(e)}"
        }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "current_model_name": current_model_name,
        "mlflow_uri": "http://mlflow:8080/"
    }


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Water Quality Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "POST /predict - Make water quality predictions",
            "update_model": "GET /update_model - Update the model version",
            "reload_model": "POST /reload_model - Reload the current model",
            "models/available": "GET /models/available - List all available models",
            "model/info": "GET /model/info - Get current model information",
            "health": "GET /health - Check API health"
        }
    }