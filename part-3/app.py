import fastapi
import pandas as pd
import numpy as np
import mlflow
import time
import requests
import random
import warnings
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterQualityPyFunc(mlflow.pyfunc.PythonModel):
    """Self-contained PyFunc wrapper for water quality models"""
    
    def __init__(self, model, scaler=None, feature_columns=None, model_type="classification"):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns or []
        self.model_type = model_type
    
    def predict(self, context, model_input):
        """Unified prediction method for all model types"""
        # Ensure input is a DataFrame
        if isinstance(model_input, np.ndarray):
            model_input = pd.DataFrame(model_input, columns=self.feature_columns)
        
        df = model_input.copy()
        
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_columns]
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(df.median())
        
        if self.scaler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_data = self.scaler.transform(df)
                result = self.model.predict(scaled_data)
                
                if self.model_type == "regression":
                    sigmoid_result = 1 / (1 + np.exp(-result))
                    result = (sigmoid_result > 0.5).astype(int)
        else:
            result = self.model.predict(df)
        
        return result

app = fastapi.FastAPI(title="Water Quality Prediction API - Canary Deployment", version="3.0.0")

# Global variables for canary deployment
current_model = None
next_model = None
current_model_version = None
next_model_version = None
current_model_name = None
next_model_name = None
canary_probability = 0.1  # 10% traffic to next model by default

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
    model_used: str  # "current" or "next"

class ModelUpdateRequest(BaseModel):
    """Request schema for model updates"""
    model_name: str = "water_quality_RandomForest"
    version: Optional[str] = "latest"

class CanaryConfigRequest(BaseModel):
    """Request schema for canary configuration"""
    probability: float

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

def load_model_from_registry(model_name: str = "water_quality_RandomForest", version: str = "latest", target: str = "current"):
    """Load model from MLflow Model Registry"""
    global current_model, next_model, current_model_version, next_model_version
    global current_model_name, next_model_name
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://mlflow:8080/")
        
        # Create experiment if it doesn't exist
        experiment_name = "water_quality_classification"
        create_experiment_if_not_exists(experiment_name)
        
        # Try to load from registry first
        try:
            if version == "latest":
                model_uri = f"models:/{model_name}/latest"
            else:
                model_uri = f"models:/{model_name}/{version}"
                
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            
            if target == "current":
                current_model = loaded_model
                current_model_version = version
                current_model_name = model_name
                logger.info(f"Successfully loaded CURRENT model {model_name} version {version}")
            elif target == "next":
                next_model = loaded_model
                next_model_version = version
                next_model_name = model_name
                logger.info(f"Successfully loaded NEXT model {model_name} version {version}")
            
            return True
            
        except Exception as registry_error:
            logger.warning(f"Failed to load from registry: {registry_error}")
            # Fall back to loading from runs
            pass
        
        # Fallback: Try to load from experiment runs
        logger.warning(f"Model '{model_name}' not found in registry. Trying fallback to experiment runs...")
        try:
            experiment = mlflow.get_experiment_by_name("water_quality_classification")
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                        order_by=["start_time DESC"], max_results=1)
                if not runs.empty:
                    run_id = runs.iloc[0]['run_id']
                    # Try multiple possible model artifact paths
                    possible_paths = [
                        f"runs:/{run_id}/random_forest_model",
                        f"runs:/{run_id}/model",
                        f"runs:/{run_id}/water_quality_model"
                    ]
                    
                    loaded_model = None
                    for model_path in possible_paths:
                        try:
                            loaded_model = mlflow.pyfunc.load_model(model_path)
                            logger.info(f"Successfully loaded model from path: {model_path}")
                            break
                        except Exception as path_error:
                            logger.warning(f"Failed to load from {model_path}: {path_error}")
                            continue
                    
                    if loaded_model is not None:
                        if target == "current":
                            current_model = loaded_model
                            current_model_version = f"run_{run_id[:8]}"
                            current_model_name = model_name
                            logger.info(f"Loaded CURRENT model from run {run_id}")
                        elif target == "next":
                            next_model = loaded_model
                            next_model_version = f"run_{run_id[:8]}"
                            next_model_name = model_name
                            logger.info(f"Loaded NEXT model from run {run_id}")
                        
                        return True
                    else:
                        logger.error("Failed to load model from any available path")
                else:
                    logger.warning(f"No runs found in experiment '{experiment_name}'")
            else:
                logger.warning(f"Experiment '{experiment_name}' not found")
        except Exception as e2:
            logger.error(f"Fallback loading also failed: {str(e2)}")
        
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load models on application startup"""
    logger.info("Starting Water Quality Prediction API with Canary Deployment...")
    
    # Wait for MLflow to be ready
    if not wait_for_mlflow():
        logger.warning("MLflow server not available - model loading skipped")
        return
    
    # Try to load the same model for both current and next at startup
    success_current = load_model_from_registry(target="current")
    success_next = load_model_from_registry(target="next")
    
    if not success_current or not success_next:
        logger.warning("Failed to load models on startup - will need to load manually")
        logger.info("You can:")
        logger.info("1. Train a model first using the training script")
        logger.info("2. Use the /update_model endpoint to load models")

def get_available_feature_columns():
    """Get the exact same feature columns that were used during training"""
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
    available_cols = get_available_feature_columns()
    
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
    
    filtered_data = {col: data_dict.get(col, 0.0) for col in available_cols}
    df = pd.DataFrame([filtered_data])
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0)
    return df

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: WaterQualityInput):
    """Canary deployment prediction endpoint - routes traffic between current and next models"""
    global current_model, next_model, canary_probability

    logger.info("Received prediction request")

    if current_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Current model not loaded. Please load models first."
        )
    

    use_next_model = next_model is not None and random.random() < canary_probability
    selected_model = next_model if use_next_model else current_model
    selected_model_version = next_model_version if use_next_model else current_model_version
    selected_model_name = next_model_name if use_next_model else current_model_name
    model_used = "next" if use_next_model else "current"
    logger.info(f"Using {model_used} model for prediction")

    try:
        df = prepare_input_data(input_data)
        logger.info(f"Input DataFrame prepared: {df}")

        prediction_result = selected_model.predict(df)
        logger.info(f"Prediction result: {prediction_result}")

        if hasattr(prediction_result, '__len__') and len(prediction_result) > 0:
            prediction = int(prediction_result[0])
        else:
            prediction = int(prediction_result)

        probability = 0.8 if prediction == 1 else 0.2

        status = "Good Water Quality" if prediction == 1 else "Bad Water Quality"
        
        logger.info(f"Prediction made using {model_used} model: {prediction}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            status=status,
            model_version=f"{selected_model_name}:{selected_model_version}" if selected_model_name else selected_model_version or "unknown",
            model_used=model_used
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
            return {
                "available_models": [],
                "total_models": 0,
                "message": "No registered models found. Train and register a model first."
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/update_model")
def update_model(request: ModelUpdateRequest):
    """Update the NEXT model (for canary deployment)"""
    try:
        success = load_model_from_registry(request.model_name, request.version, target="next")
        if success:
            return {
                "message": f"Successfully updated NEXT model to {request.model_name} version {request.version}",
                "next_model_version": next_model_version,
                "next_model_name": next_model_name,
                "canary_probability": canary_probability
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Failed to load model '{request.model_name}' version '{request.version}'. Make sure the model exists in MLflow registry or experiment runs."
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model update failed: {str(e)}")

@app.post("/accept_next_model")
def accept_next_model():
    """Accept the next model as the current model (promote canary deployment)"""
    global current_model, next_model, current_model_version, next_model_version
    global current_model_name, next_model_name
    
    if next_model is None:
        raise HTTPException(
            status_code=400,
            detail="No next model loaded. Load a next model first using /update_model."
        )
    
    try:
        # Promote next model to current
        current_model = next_model
        current_model_version = next_model_version
        current_model_name = next_model_name
        
        logger.info(f"Promoted next model to current: {current_model_name} version {current_model_version}")
        
        return {
            "message": "Next model successfully promoted to current model",
            "current_model": f"{current_model_name}:{current_model_version}",
            "next_model": f"{next_model_name}:{next_model_version}",
            "canary_probability": canary_probability
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model promotion failed: {str(e)}")

@app.post("/canary_config")
def set_canary_probability(request: CanaryConfigRequest):
    """Set the canary deployment probability (0.0 = all current, 1.0 = all next)"""
    global canary_probability
    
    if not 0.0 <= request.probability <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Probability must be between 0.0 and 1.0"
        )
    
    canary_probability = request.probability
    
    return {
        "message": f"Canary probability set to {canary_probability}",
        "current_traffic_percentage": round((1 - canary_probability) * 100, 1),
        "next_traffic_percentage": round(canary_probability * 100, 1)
    }

@app.get("/canary_status")
def get_canary_status():
    """Get the current canary deployment status"""
    return {
        "current_model": {
            "name": current_model_name,
            "version": current_model_version,
            "loaded": current_model is not None,
            "traffic_percentage": round((1 - canary_probability) * 100, 1)
        },
        "next_model": {
            "name": next_model_name,
            "version": next_model_version,
            "loaded": next_model is not None,
            "traffic_percentage": round(canary_probability * 100, 1)
        },
        "canary_probability": canary_probability
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "current_model_loaded": current_model is not None,
        "next_model_loaded": next_model is not None,
        "canary_probability": canary_probability,
        "mlflow_uri": "http://mlflow:8080/"
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Water Quality Prediction API - Canary Deployment",
        "version": "3.0.0",
        "status": "running",
        "canary_deployment": {
            "current_model_loaded": current_model is not None,
            "next_model_loaded": next_model is not None,
            "canary_probability": canary_probability
        },
        "endpoints": {
            "predict": "POST /predict - Make water quality predictions (canary deployment)",
            "update_model": "POST /update_model - Update the next model for canary deployment",
            "accept_next_model": "POST /accept_next_model - Promote next model to current",
            "canary_config": "POST /canary_config - Set canary deployment probability",
            "canary_status": "GET /canary_status - Get current canary deployment status",
            "models/available": "GET /models/available - List all available models",
            "health": "GET /health - Check API health"
        }
    }