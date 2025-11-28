"""
Test script for the deployed model API
=======================================

This script demonstrates how to call the deployed model API endpoints.
"""

import requests
import json

# API base URL
# BASE_URL = "http://localhost:5000"  # For Flask
BASE_URL = "http://localhost:8000"  # For FastAPI

def test_health_check():
    """Test the health check endpoint"""
    print("="*80)
    print("Testing Health Check Endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_single_prediction():
    """Test single prediction endpoint"""
    print("="*80)
    print("Testing Single Prediction Endpoint")
    print("="*80)
    
    # Sample product data
    product_data = {
        "country": "Canada",
        "entity": "PRO STANDARD CA",
        "currency": "CAD",
        "fin_sku": "BTR6515792-BLK-XL",
        "fin_brand": "Pro Standard",
        "fin_team_name": "TORONTO RAPTORS",
        "fin_department": "Mens",
        "fin_gender": "Male",
        "fin_class": "Tops",
        "fin_league_license": "NBA",
        "location": "Warehouse A",
        "fpo": 10.0,
        "stock": 50.0,
        "open_orders": 20.0,
        "avail_now": 30.0,
        "ordered_qty": 5.0,
        "shipped_qty": 3.0,
        "open_qty": 2.0,
        "avail_later": 10.0,
        "priority": 1,
        "latest_po_received_date": "2025-11-15",
        "stock_age_range": "(1) 0-2m",
        "pf_type": "Standard"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=product_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("="*80)
    print("Testing Batch Prediction Endpoint")
    print("="*80)
    
    # Sample batch data
    batch_data = {
        "data": [
            {
                "country": "Canada",
                "entity": "PRO STANDARD CA",
                "currency": "CAD",
                "fin_sku": "BTR6515792-BLK-XL",
                "fin_brand": "Pro Standard",
                "fin_team_name": "TORONTO RAPTORS",
                "fin_department": "Mens",
                "fin_gender": "Male",
                "fin_class": "Tops",
                "fin_league_license": "NBA",
                "location": "Warehouse A",
                "fpo": 10.0,
                "stock": 50.0,
                "open_orders": 20.0,
                "avail_now": 30.0,
                "ordered_qty": 5.0,
                "shipped_qty": 3.0,
                "open_qty": 2.0,
                "avail_later": 10.0,
                "priority": 1,
                "latest_po_received_date": "2025-11-15",
                "stock_age_range": "(1) 0-2m",
                "pf_type": "Standard"
            },
            {
                "country": "USA",
                "entity": "PRO STANDARD US",
                "currency": "USD",
                "fin_sku": "HWJ567470-MDN-S",
                "fin_brand": "Pro Standard",
                "fin_team_name": "WINNIPEG JETS",
                "fin_department": "Mens",
                "fin_gender": "Male",
                "fin_class": "Tops",
                "fin_league_license": "NHL",
                "location": "Warehouse B",
                "fpo": 15.0,
                "stock": 100.0,
                "open_orders": 50.0,
                "avail_now": 75.0,
                "ordered_qty": 10.0,
                "shipped_qty": 8.0,
                "open_qty": 2.0,
                "avail_later": 25.0,
                "priority": 2,
                "latest_po_received_date": "2025-11-20",
                "stock_age_range": "(2) 4-6m",
                "pf_type": "Premium"
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=batch_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_model_info():
    """Test model info endpoint"""
    print("="*80)
    print("Testing Model Info Endpoint")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("API TEST SUITE - DEMAND PREDICTION MODEL")
    print("="*80)
    print(f"\nTesting API at: {BASE_URL}")
    print("\nMake sure the API server is running before executing these tests!")
    print("  Flask:   python flask_api.py")
    print("  FastAPI: uvicorn fastapi_api:app --reload")
    print("\n")
    
    try:
        # Run all tests
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("="*80)
        print("✓ All tests completed successfully!")
        print("="*80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server!")
        print(f"   Make sure the server is running at {BASE_URL}")
        print("\nTo start the server:")
        print("  Flask:   python flask_api.py")
        print("  FastAPI: uvicorn fastapi_api:app --reload")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
