"""
Flask REST API for Demand Prediction Model
===========================================

This script creates a REST API using Flask to serve the trained model.

To run:
    python flask_api.py

API will be available at: http://localhost:5000
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the predictor class
from deploy_model import DemandPredictor

# Initialize Flask app
app = Flask(__name__)

# Load model on startup
print("Loading model...")
predictor = DemandPredictor('best_model_fixed.pkl')
print("✓ Model loaded successfully!")


@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Sports Merchandise Demand Prediction API',
        'model': predictor.model_name,
        'accuracy': float(predictor.metrics['Accuracy']),
        'f1_score': float(predictor.metrics['F1-Score']),
        'endpoints': {
            '/': 'GET - API documentation',
            '/health': 'GET - Health check',
            '/predict': 'POST - Single prediction',
            '/predict/batch': 'POST - Batch prediction',
            '/model/info': 'GET - Model information'
        },
        'example_request': {
            'country': 'Canada',
            'entity': 'PRO STANDARD CA',
            'currency': 'CAD',
            'fin_sku': 'BTR6515792-BLK-XL',
            'fin_brand': 'Pro Standard',
            'fin_team_name': 'TORONTO RAPTORS',
            'fin_department': 'Mens',
            'fin_gender': 'Male',
            'fin_class': 'Tops',
            'fin_league_license': 'NBA',
            'location': 'Warehouse A',
            'fpo': 10.0,
            'stock': 50.0,
            'open_orders': 20.0,
            'avail_now': 30.0,
            'ordered_qty': 5.0,
            'shipped_qty': 3.0,
            'open_qty': 2.0,
            'avail_later': 10.0,
            'priority': 1,
            'latest_po_received_date': '2025-11-15',
            'stock_age_range': '(1) 0-2m',
            'pf_type': 'Standard'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': predictor.model_name,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Example request:
    {
        "country": "Canada",
        "entity": "PRO STANDARD CA",
        "currency": "CAD",
        "fin_sku": "BTR6515792-BLK-XL",
        ...
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = predictor.predict(data)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Example request:
    {
        "data": [
            {"country": "Canada", "entity": "PRO STANDARD CA", ...},
            {"country": "USA", "entity": "PRO STANDARD US", ...}
        ]
    }
    """
    try:
        # Get JSON data from request
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'No data array provided'}), 400
        
        data = request_data['data']
        
        if not isinstance(data, list):
            return jsonify({'error': 'Data must be an array'}), 400
        
        # Make predictions
        results = predictor.predict(pd.DataFrame(data))
        
        return jsonify({
            'success': True,
            'count': len(results) if isinstance(results, list) else 1,
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model/info')
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': predictor.model_name,
        'metrics': {
            'accuracy': float(predictor.metrics['Accuracy']),
            'precision': float(predictor.metrics['Precision']),
            'recall': float(predictor.metrics['Recall']),
            'f1_score': float(predictor.metrics['F1-Score']),
            'roc_auc': float(predictor.metrics['ROC-AUC']) if predictor.metrics['ROC-AUC'] else None
        },
        'features': {
            'count': len(predictor.feature_names),
            'names': predictor.feature_names
        },
        'target_variable': predictor.model_package.get('target_variable', 'high_order_demand'),
        'target_description': predictor.model_package.get('target_description', ''),
        'data_leakage_fixed': predictor.model_package.get('data_leakage_fixed', False),
        'removed_features': predictor.model_package.get('removed_features', [])
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FLASK API SERVER - DEMAND PREDICTION MODEL")
    print("="*80)
    print(f"\nModel: {predictor.model_name}")
    print(f"Accuracy: {predictor.metrics['Accuracy']:.4f}")
    print(f"F1-Score: {predictor.metrics['F1-Score']:.4f}")
    print(f"\nStarting server on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /              - API documentation")
    print("  GET  /health        - Health check")
    print("  POST /predict       - Single prediction")
    print("  POST /predict/batch - Batch prediction")
    print("  GET  /model/info    - Model information")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
