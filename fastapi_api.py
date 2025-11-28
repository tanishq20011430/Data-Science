"""
FastAPI REST API for Demand Prediction Model
============================================

Modern, high-performance API using FastAPI with automatic documentation.

To run:
    uvicorn fastapi_api:app --reload

API will be available at: http://localhost:8000
Documentation at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the predictor class
from deploy_model import DemandPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Sports Merchandise Demand Prediction API",
    description="API for predicting high vs low order demand for sports merchandise products",
    version="1.0.0"
)

# Load model on startup
predictor = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global predictor
    print("Loading model...")
    predictor = DemandPredictor('best_model_fixed.pkl')
    print("✓ Model loaded successfully!")


# Pydantic models for request/response validation
class ProductData(BaseModel):
    """Product data for prediction"""
    country: Optional[str] = None
    entity: Optional[str] = None
    currency: Optional[str] = None
    fin_sku: Optional[str] = None
    fin_brand: Optional[str] = None
    fin_team_name: Optional[str] = None
    fin_department: Optional[str] = None
    fin_gender: Optional[str] = None
    fin_class: Optional[str] = None
    fin_league_license: Optional[str] = None
    location: Optional[str] = None
    fpo: Optional[float] = 0.0
    stock: Optional[float] = 0.0
    open_orders: Optional[float] = 0.0
    avail_now: Optional[float] = 0.0
    ordered_qty: Optional[float] = 0.0
    shipped_qty: Optional[float] = 0.0
    open_qty: Optional[float] = 0.0
    avail_later: Optional[float] = 0.0
    priority: Optional[int] = 1
    latest_po_received_date: Optional[str] = None
    stock_age_range: Optional[str] = None
    pf_type: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[ProductData]


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: str
    prediction_value: int
    probability_low_demand: Optional[float] = None
    probability_high_demand: Optional[float] = None
    confidence: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    success: bool
    count: int
    predictions: List[PredictionResponse]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    metrics: Dict[str, Any]
    features: Dict[str, Any]
    target_variable: str
    target_description: str
    data_leakage_fixed: bool
    removed_features: List[str]


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sports Merchandise Demand Prediction API",
        "model": predictor.model_name,
        "accuracy": float(predictor.metrics['Accuracy']),
        "f1_score": float(predictor.metrics['F1-Score']),
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "POST /predict": "Single product prediction",
            "POST /predict/batch": "Batch prediction",
            "GET /model/info": "Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": predictor.model_name if predictor else "not loaded",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(product: ProductData):
    """
    Make a prediction for a single product
    
    Returns prediction with probabilities for high vs low demand
    """
    try:
        # Convert to dict and make prediction
        data = product.dict()
        result = predictor.predict(data)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple products
    
    Returns predictions with probabilities for each product
    """
    try:
        # Convert to list of dicts
        data = [product.dict() for product in request.data]
        
        # Make predictions
        results = predictor.predict(pd.DataFrame(data))
        
        # Ensure results is a list
        if not isinstance(results, list):
            results = [results]
        
        return {
            "success": True,
            "count": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get detailed model information"""
    return {
        "model_name": predictor.model_name,
        "metrics": {
            "accuracy": float(predictor.metrics['Accuracy']),
            "precision": float(predictor.metrics['Precision']),
            "recall": float(predictor.metrics['Recall']),
            "f1_score": float(predictor.metrics['F1-Score']),
            "roc_auc": float(predictor.metrics['ROC-AUC']) if predictor.metrics['ROC-AUC'] else None,
            "training_time": float(predictor.metrics.get('Training Time (s)', 0))
        },
        "features": {
            "count": len(predictor.feature_names),
            "names": predictor.feature_names
        },
        "target_variable": predictor.model_package.get('target_variable', 'high_order_demand'),
        "target_description": predictor.model_package.get('target_description', 'Predicts if product will have high order demand'),
        "data_leakage_fixed": predictor.model_package.get('data_leakage_fixed', False),
        "removed_features": predictor.model_package.get('removed_features', [])
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("FASTAPI SERVER - DEMAND PREDICTION MODEL")
    print("="*80)
    print("\nStarting server on http://localhost:8000")
    print("\nAvailable endpoints:")
    print("  GET  /              - API information")
    print("  GET  /health        - Health check")
    print("  POST /predict       - Single prediction")
    print("  POST /predict/batch - Batch prediction")
    print("  GET  /model/info    - Model information")
    print("  GET  /docs          - Interactive API documentation")
    print("  GET  /redoc         - Alternative API documentation")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
