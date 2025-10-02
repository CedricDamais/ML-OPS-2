#!/usr/bin/env python3
"""
Quick test to verify LinearRegression predictions work without hanging
"""

import requests
import json
import time

def test_linear_regression_fix():
    """Test that LinearRegression predictions work without hanging"""
    
    base_url = "http://localhost:8000"  # Update port as needed
    
    print("🔧 Testing LinearRegression Prediction Fix")
    print("=" * 50)
    
    # Test data
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
    
    # 1. Check current model
    print("1. Checking current model...")
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Current model: {info['current_model_name']} (v{info['model_version']})")
        else:
            print(f"❌ Failed to get model info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 2. Switch to LinearRegression
    print("\n2. Switching to LinearRegression...")
    try:
        response = requests.get(f"{base_url}/update_model", 
                              params={"model_name": "water_quality_LinearRegression", "version": "latest"},
                              timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
        else:
            print(f"❌ Failed to switch: {response.status_code} - {response.text}")
            return
    except Exception as e:
        print(f"❌ Error switching: {e}")
        return
    
    # 3. Test prediction with timeout to detect hanging
    print("\n3. Testing LinearRegression prediction (with timeout)...")
    start_time = time.time()
    try:
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=15)
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful in {duration:.2f}s")
            print(f"   Result: {result['prediction']} ({result['status']})")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Model: {result['model_version']}")
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT: Prediction took longer than 15s - still hanging!")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 4. Switch back to RandomForest for comparison
    print("\n4. Switching back to RandomForest...")
    try:
        response = requests.get(f"{base_url}/update_model", 
                              params={"model_name": "water_quality_RandomForest", "version": "latest"},
                              timeout=10)
        if response.status_code == 200:
            print("✅ Switched back to RandomForest")
        else:
            print(f"❌ Failed to switch back: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # 5. Test RandomForest prediction
    print("\n5. Testing RandomForest prediction...")
    try:
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ RandomForest prediction: {result['prediction']} ({result['status']})")
            print(f"   Probability: {result['probability']:.4f}")
        else:
            print(f"❌ RandomForest prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ LinearRegression hanging test completed!")

if __name__ == "__main__":
    test_linear_regression_fix()