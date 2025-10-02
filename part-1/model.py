import mlflow
from mlflow.models import infer_signature

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler


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

PORT = '8080'
HOST = 'localhost'
TRACKING_URL = f'http://{HOST}:{PORT}/'

class MLFlowLogger:
    def __init__(self, experiment_name: str, run_name: str, tracking_uri: str = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        self.run_id = self.run.info.run_id
        self.model_info = None

    def get_run(self):
        return mlflow.get_run(self.run_id)
    
    def get_model_info(self):
        return self.model_info

    def set_model_info(self, model_info):
        self.model_info = model_info
    
    def log_params(self, params):
        """Log parameters to MLflow"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """Log metrics to MLflow"""
        for metric_name, metric_value in metrics.items():
            # Only log numeric values as metrics
            if metric_name not in ['classification_report', 'model_type'] and isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
    
    def log_model(self, model, model_name, signature=None, input_example=None, registered_model_name=None):
        """Log model to MLflow using PyFunc"""
        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )
        return model_info
    
    def log_text(self, text, filename):
        """Log text file to MLflow"""
        mlflow.log_text(text, filename)
    
    def set_tags(self, tags):
        """Set tags for the current MLflow run"""
        if isinstance(tags, list):
            tag_dict = {f"tag_{i}": tag for i, tag in enumerate(tags)}
        elif isinstance(tags, dict):
            tag_dict = tags
        else:
            tag_dict = {"tag_0": str(tags)}
        
        mlflow.set_tags(tag_dict)
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()

    def load_model(self):
        """Load a model from MLflow"""
        return mlflow.pyfunc.load_model(self.model_info.model_uri)


def load_data():
    df = pd.read_csv("../data/Indian_water_data.csv")
    return df


def select_numerical_features(df):
    """Select only numerical columns for features"""

    numerical_cols = [
        'Temperature (C) - Min', 'Temperature (C) - Max',
        'Dissolved - Min', 'Dissolved - Max',
        'pH - Min', 'pH - Max',
        'Conductivity (µmho/cm) - Min', 'Conductivity (µmho/cm) - Max',
        'BOD (mg/L) - Min', 'BOD (mg/L) - Max',
        'NitrateN (mg/L) - Min', 'NitrateN (mg/L) - Max',
        'Fecal Coliform (MPN/100ml) - Min', 'Fecal Coliform (MPN/100ml) - Max',
        'Total Coliform (MPN/100ml) - Min', 'Total Coliform (MPN/100ml) - Max',
        'Fecal - Min', 'Fecal - Max'
    ]
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    X = df[available_cols].copy()
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(X.median())
    
    return X, available_cols


def train_classification_model(X, y):
    """Train a Random Forest classifier with DataFrame input to preserve feature names"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest doesn't necessarily need scaling, so let's train directly on DataFrames
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create PyFunc wrapper
    pyfunc_model = WaterQualityPyFunc(model, None, list(X.columns), "classification")
    
    return pyfunc_model, None, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(y_test, y_pred)
    }, X_test, y_test, y_pred

def train_linear_regression_model(X, y):
    """Train a Linear Regression model with PyFunc wrapper"""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    pyfunc_model = WaterQualityPyFunc(model, scaler, list(X.columns), "regression")
    
    return pyfunc_model, scaler, {
        'mse': mse,
        'r2_score': r2
    }, X_test, y_test, y_pred

def log_experiment(experiment_name, run_name, tracking_uri, params, metrics, model=None, scaler=None, X_train=None, tag_names=None, registered_model_name=None):
    """Log experiment parameters, metrics, and model to MLflow using custom MLFlowLogger"""
    logger = MLFlowLogger(experiment_name, run_name, tracking_uri)
    
    try:
        logger.log_params(params)
        
        logger.log_metrics(metrics)
        
        if 'classification_report' in metrics:
            logger.log_text(metrics['classification_report'], "classification_report.txt")

        if model is not None:
            # Use DataFrame for signature to maintain feature names consistency with API
            # For PyFunc models, we need to predict with the model directly to get sample output
            sample_prediction = model.predict(None, X_train.iloc[:1])
            signature = infer_signature(X_train, sample_prediction)
            model_info = logger.log_model(model, "model", 
                            signature=signature, 
                            input_example=X_train.iloc[:5], 
                            registered_model_name=registered_model_name)
            logger.set_model_info(model_info)
            logger.set_tags(tag_names)


        print("FINISHED LOGGING TO MLFLOW")
        print(f"Run ID: {logger.run_id}")
        
    finally:
        logger.end_run()
    
    return logger


def run_experiment():
    """Run the complete ML pipeline with MLflow tracking"""
    df = load_data()
    
    df['Status'] = ((df['Temperature (C) - Max'] > 25) & (df['pH - Min'] > 6.5))
    df['Status'] = df['Status'].astype(int)
    
    X, feature_cols = select_numerical_features(df)
    y = df['Status']
    
    params = {
        "n_estimators": 10,
        "random_state": 42,
        "test_size": 0.2,
        "n_features": len(feature_cols),
        "n_samples": len(df),
        "model_type": "RandomForestClassifier"
    }
    
    model, scaler, metrics, X_test, y_test, y_pred = train_classification_model(X, y)
    
    logger = log_experiment(
        experiment_name="water_quality_classification",
        run_name="random_forest_experiment", 
        tracking_uri=TRACKING_URL,
        params=params,
        metrics=metrics,
        model=model,
        scaler=scaler,
        X_train=X,  # Pass training data for signature inference
        tag_names=["v1"],
        registered_model_name="water_quality_RandomForest"
    )

    lr_model, lr_scaler, lr_metrics, X_test, y_test, y_pred = train_linear_regression_model(X, y)

    log_experiment(
        experiment_name="water_quality_classification",
        run_name="linear_regression_experiment", 
        tracking_uri=TRACKING_URL,
        params={
            "test_size": 0.2,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "model_type": "LinearRegression"
        },
        metrics=lr_metrics,
        model=lr_model,
        scaler=None,
        X_train=X,
        tag_names=["LR-v1"],
        registered_model_name="water_quality_LinearRegression"
    )
    
    print(f"\nLinear Regression Performance:")
    print(f"MSE: {lr_metrics['mse']:.4f}")
    print(f"R² Score: {lr_metrics['r2_score']:.4f}")
    
    print(f"\nRandom Forest Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])

    return model, scaler, metrics, logger


if __name__ == "__main__":
    model, scaler, metrics, logger = run_experiment()
