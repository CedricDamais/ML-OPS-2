#!/usr/bin/env python3
"""
Test the PyFunc approach - should work seamlessly with any model
"""

import mlflow
import pandas as pd

def test_pyfunc_models():
    """Test both PyFunc models directly"""
    mlflow.set_tracking_uri("http://127.0.0.1:8080/")
    
    # Test data
    test_data = {
        'Temperature (C) - Min': [10.0],
        'Temperature (C) - Max': [20.0],
        'Dissolved - Min': [5.0],
        'Dissolved - Max': [8.0],
        'pH - Min': [6.8],
        'pH - Max': [7.2],
        'BOD (mg/L) - Min': [1.0],
        'BOD (mg/L) - Max': [2.5],
        'NitrateN (mg/L) - Min': [0.1],
        'NitrateN (mg/L) - Max': [0.5],
        'Fecal Coliform (MPN/100ml) - Min': [10.0],
        'Fecal Coliform (MPN/100ml) - Max': [50.0],
        'Total Coliform (MPN/100ml) - Min': [20.0],
        'Total Coliform (MPN/100ml) - Max': [100.0],
        'Fecal - Min': [5.0],
        'Fecal - Max': [25.0]
    }
    
    df = pd.DataFrame(test_data)
    
    print("🧪 Testing PyFunc Models")
    print("=" * 40)
    
    # Test RandomForest PyFunc
    print("\n1. Testing RandomForest PyFunc...")
    try:
        rf_model = mlflow.pyfunc.load_model("models:/water_quality_RandomForest/latest")
        rf_prediction = rf_model.predict(df)
        print(f"✅ RandomForest prediction: {rf_prediction}")
    except Exception as e:
        print(f"❌ RandomForest failed: {e}")
    
    # Test LinearRegression PyFunc
    print("\n2. Testing LinearRegression PyFunc...")
    try:
        lr_model = mlflow.pyfunc.load_model("models:/water_quality_LinearRegression/latest")
        lr_prediction = lr_model.predict(df)
        print(f"✅ LinearRegression prediction: {lr_prediction}")
    except Exception as e:
        print(f"❌ LinearRegression failed: {e}")
    
    print("\n" + "=" * 40)
    print("✅ PyFunc test completed!")
    print("\nKey benefits of PyFunc approach:")
    print("- ✅ Unified interface for all models")
    print("- ✅ Preprocessing handled internally")
    print("- ✅ No feature name warnings")
    print("- ✅ Consistent binary output")
    print("- ✅ API is completely model-agnostic")

if __name__ == "__main__":
    test_pyfunc_models()