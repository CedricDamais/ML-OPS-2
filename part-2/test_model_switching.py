#!/usr/bin/env python3
"""
Test script to demonstrate switching between RandomForest and LinearRegression models
"""

import requests
import json

def test_model_switching():
    """Test switching between different models"""
    
    base_url = "http://localhost:8003"  # Update port as needed
    
    print("🔍 Testing Model Registry Features")
    print("=" * 50)
    
    # 1. List available models
    print("\n1. Listing available models...")
    try:
        response = requests.get(f"{base_url}/models/available")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Found {models['total_models']} models:")
            for model in models['available_models']:
                print(f"   - {model['name']}: versions {model['versions']}")
        else:
            print(f"❌ Failed to list models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    
    # 2. Check current model info
    print("\n2. Checking current model info...")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Current model version: {info['model_version']}")
            print(f"   Available models: {info['available_models']}")
        else:
            print(f"❌ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
    
    # 3. Test prediction with RandomForest (default)
    print("\n3. Testing prediction with RandomForest...")
    test_data = {
        "temperature_min": 10.0,
        "temperature_max": 20.0,
        "dissolved_min": 5.0,
        "dissolved_max": 8.0,
        "ph_min": 6.8,
        "ph_max": 7.2,
        "bod_min": 1.0,
        "bod_max": 2.5,
        "nitrate_min": 0.1,
        "nitrate_max": 0.5,
        "fecal_coliform_min": 10.0,
        "fecal_coliform_max": 50.0,
        "total_coliform_min": 20.0,
        "total_coliform_max": 100.0,
        "fecal_min": 5.0,
        "fecal_max": 25.0
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ RandomForest prediction: {result['prediction']} (Quality: {result['quality_description']})")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
    
    # 4. Switch to LinearRegression model
    print("\n4. Switching to LinearRegression model...")
    try:
        response = requests.get(f"{base_url}/update_model", 
                              params={"model_name": "water_quality_LinearRegression", "version": "latest"})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Switched to: {result['message']}")
        else:
            print(f"❌ Failed to switch model: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error switching model: {e}")
    
    # 5. Test prediction with LinearRegression
    print("\n5. Testing prediction with LinearRegression...")
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ LinearRegression prediction: {result['prediction']}")
            print(f"   Note: Linear regression returns continuous values, not quality classes")
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
    
    # 6. Switch back to RandomForest
    print("\n6. Switching back to RandomForest...")
    try:
        response = requests.get(f"{base_url}/update_model", 
                              params={"model_name": "water_quality_RandomForest", "version": "latest"})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Switched back to: {result['message']}")
        else:
            print(f"❌ Failed to switch model: {response.status_code}")
    except Exception as e:
        print(f"❌ Error switching model: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Model switching test completed!")
    print("\nKey Features Demonstrated:")
    print("- ✅ List all available models in MLflow registry")
    print("- ✅ Get current model information")
    print("- ✅ Switch between RandomForest and LinearRegression models")
    print("- ✅ Make predictions with different model types")
    print("- ✅ Proper error handling for model operations")

if __name__ == "__main__":
    test_model_switching()