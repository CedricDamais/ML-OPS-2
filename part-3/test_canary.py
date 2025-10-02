#!/usr/bin/env python3
"""
Test script for Canary Deployment API
"""

import requests
import json
import time
import random

API_BASE = "http://localhost:8001"

def test_canary_deployment():
    """Test the canary deployment functionality"""
    
    print("🚀 Testing Canary Deployment API")
    print("=" * 50)
    
    # 1. Check API status
    print("\n1. Checking API status...")
    response = requests.get(f"{API_BASE}/")
    print(f"API Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Current model loaded: {data['canary_deployment']['current_model_loaded']}")
        print(f"Next model loaded: {data['canary_deployment']['next_model_loaded']}")
        print(f"Canary probability: {data['canary_deployment']['canary_probability']}")
    
    # 2. Check canary status
    print("\n2. Checking canary status...")
    response = requests.get(f"{API_BASE}/canary_status")
    if response.status_code == 200:
        status = response.json()
        print(f"Current model: {status['current_model']['name']} v{status['current_model']['version']}")
        print(f"Next model: {status['next_model']['name']} v{status['next_model']['version']}")
        print(f"Traffic split - Current: {status['current_model']['traffic_percentage']}%, Next: {status['next_model']['traffic_percentage']}%")
    
    # 3. Test predictions with different canary probabilities
    print("\n3. Testing predictions with different canary probabilities...")
    
    # Sample input data
    test_input = {
        "temperature_min": 20.0,
        "temperature_max": 25.0,
        "dissolved_min": 5.0,
        "dissolved_max": 8.0,
        "ph_min": 6.5,
        "ph_max": 7.5,
        "bod_min": 2.0,
        "bod_max": 4.0,
        "nitrate_min": 1.0,
        "nitrate_max": 3.0,
        "fecal_coliform_min": 10.0,
        "fecal_coliform_max": 50.0,
        "total_coliform_min": 20.0,
        "total_coliform_max": 100.0,
        "fecal_min": 5.0,
        "fecal_max": 15.0
    }
    
    # Test with 10% canary traffic
    print("\n   a) Testing with 10% canary traffic...")
    requests.post(f"{API_BASE}/canary_config", json={"probability": 0.1})
    
    current_count = 0
    next_count = 0
    
    for i in range(20):
        response = requests.post(f"{API_BASE}/predict", json=test_input)
        if response.status_code == 200:
            result = response.json()
            if result["model_used"] == "current":
                current_count += 1
            else:
                next_count += 1
    
    print(f"   Results: Current model used {current_count} times, Next model used {next_count} times")
    
    # Test with 50% canary traffic
    print("\n   b) Testing with 50% canary traffic...")
    requests.post(f"{API_BASE}/canary_config", json={"probability": 0.5})
    
    current_count = 0
    next_count = 0
    
    for i in range(20):
        response = requests.post(f"{API_BASE}/predict", json=test_input)
        if response.status_code == 200:
            result = response.json()
            if result["model_used"] == "current":
                current_count += 1
            else:
                next_count += 1
    
    print(f"   Results: Current model used {current_count} times, Next model used {next_count} times")
    
    # 4. Test model promotion
    print("\n4. Testing model promotion...")
    response = requests.post(f"{API_BASE}/accept_next_model")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"Current model: {result['current_model']}")
    else:
        print(f"❌ Model promotion failed: {response.text}")
    
    # Reset canary probability
    requests.post(f"{API_BASE}/canary_config", json={"probability": 0.1})
    
    print("\n🎉 Canary deployment test completed!")

if __name__ == "__main__":
    try:
        test_canary_deployment()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the canary app is running on port 8001")
    except Exception as e:
        print(f"❌ Error during testing: {e}")