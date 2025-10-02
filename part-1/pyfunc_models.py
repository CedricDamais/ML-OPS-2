#!/usr/bin/env python3
"""
Custom PyFunc model wrappers for water quality prediction
"""

import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings


class WaterQualityClassifierPyFunc(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapper for water quality classification models"""
    
    def __init__(self, model, feature_columns, model_type="classification"):
        self.model = model
        self.feature_columns = feature_columns
        self.model_type = model_type
        self.scaler = None
    
    def set_scaler(self, scaler):
        """Set the scaler for preprocessing"""
        self.scaler = scaler
    
    def predict(self, context, model_input):
        """Predict using the wrapped model with proper preprocessing"""
        # Ensure input is a DataFrame
        if isinstance(model_input, np.ndarray):
            model_input = pd.DataFrame(model_input, columns=self.feature_columns)
        
        # Preprocessing
        df = model_input.copy()
        
        # Ensure we have the right columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Reorder columns to match training
        df = df[self.feature_columns]
        
        # Convert to numeric and fill NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(df.median())
        
        # Apply scaling if needed
        if self.scaler is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled_data = self.scaler.transform(df)
                result = self.model.predict(scaled_data)
        else:
            # No scaling needed (e.g., for RandomForest)
            result = self.model.predict(df)
        
        return result


class WaterQualityRegressorPyFunc(mlflow.pyfunc.PythonModel):
    """Custom PyFunc wrapper for water quality regression models"""
    
    def __init__(self, model, scaler, feature_columns):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
    
    def predict(self, context, model_input):
        """Predict using the wrapped regression model with proper preprocessing"""
        # Ensure input is a DataFrame
        if isinstance(model_input, np.ndarray):
            model_input = pd.DataFrame(model_input, columns=self.feature_columns)
        
        # Preprocessing
        df = model_input.copy()
        
        # Ensure we have the right columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Reorder columns to match training
        df = df[self.feature_columns]
        
        # Convert to numeric and fill NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(df.median())
        
        # Apply scaling and suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_data = self.scaler.transform(df)
            continuous_result = self.model.predict(scaled_data)
        
        # Convert regression output to binary classification
        # Apply sigmoid to map to [0,1] range, then threshold at 0.5
        sigmoid_result = 1 / (1 + np.exp(-continuous_result))
        binary_result = (sigmoid_result > 0.5).astype(int)
        
        return binary_result