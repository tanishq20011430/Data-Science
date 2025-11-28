# Model Deployment Guide

## 📦 Deployed Model: Sports Merchandise Demand Prediction

This guide provides everything you need to deploy and use the trained machine learning model for predicting high vs low order demand for sports merchandise products.

---

## 🎯 Model Overview

- **Model Type**: Logistic Regression (best performer)
- **Accuracy**: ~100% (with data leakage) → ~77-80% (after fixing leakage)
- **Task**: Binary classification (High Order Demand vs Low Order Demand)
- **Target**: Products with `ordered_qty > 75th percentile`
- **Features**: 26-27 clean features (no data leakage)

---

## 📁 Files Included

### Core Deployment Files
1. **`best_model_fixed.pkl`** - Trained model with metadata
2. **`deploy_model.py`** - Python class for model predictions
3. **`flask_api.py`** - Flask REST API server
4. **`fastapi_api.py`** - FastAPI server (modern alternative)
5. **`test_api.py`** - API testing script
6. **`DEPLOYMENT_GUIDE.md`** - This file

---

## 🚀 Quick Start

### Option 1: Direct Python Usage

```python
from deploy_model import DemandPredictor

# Load the model
predictor = DemandPredictor('best_model_fixed.pkl')

# Make a prediction
product_data = {
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

result = predictor.predict(product_data)
print(result)
# Output: {'prediction': 'High Order Demand', 'probability_high_demand': 0.85, ...}
```

### Option 2: Flask REST API

```bash
# Install dependencies
pip install flask pandas numpy scikit-learn xgboost

# Start the server
python flask_api.py

# API available at: http://localhost:5000
```

### Option 3: FastAPI (Recommended for Production)

```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn xgboost

# Start the server
uvicorn fastapi_api:app --reload

# API available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

---

## 🔌 API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "Logistic Regression",
  "timestamp": "2025-11-29T10:30:00"
}
```

### 2. Single Prediction
```bash
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "country": "Canada",
  "fin_sku": "BTR6515792-BLK-XL",
  "stock": 50.0,
  "open_orders": 20.0,
  "avail_now": 30.0,
  ...
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "prediction": "High Order Demand",
    "prediction_value": 1,
    "probability_low_demand": 0.15,
    "probability_high_demand": 0.85,
    "confidence": 0.85
  },
  "timestamp": "2025-11-29T10:30:00"
}
```

### 3. Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json
```

**Request:**
```json
{
  "data": [
    {"country": "Canada", "stock": 50, ...},
    {"country": "USA", "stock": 100, ...}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "count": 2,
  "predictions": [
    {"prediction": "High Order Demand", ...},
    {"prediction": "Low Order Demand", ...}
  ],
  "timestamp": "2025-11-29T10:30:00"
}
```

### 4. Model Information
```bash
GET /model/info
```

**Response:**
```json
{
  "model_name": "Logistic Regression",
  "metrics": {
    "accuracy": 0.7744,
    "precision": 0.7500,
    "recall": 0.8000,
    "f1_score": 0.7741,
    "roc_auc": 0.8500
  },
  "features": {
    "count": 27,
    "names": ["fpo", "stock", "open_orders", ...]
  },
  "target_variable": "high_order_demand",
  "data_leakage_fixed": true,
  "removed_features": ["fulfillment_rate", "inventory_utilization", "ordered_qty"]
}
```

---

## 🧪 Testing the API

```bash
# Install requests library
pip install requests

# Run the test script
python test_api.py
```

Or use curl:

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "Canada", "stock": 50, "open_orders": 20}'

# Model info
curl http://localhost:5000/model/info
```

---

## 📊 Required Features

The model expects the following features (some are auto-generated):

### Raw Input Features
- `country`, `entity`, `currency`
- `fin_sku`, `fin_brand`, `fin_team_name`
- `fin_department`, `fin_gender`, `fin_class`, `fin_league_license`
- `location`, `pf_type`
- `fpo`, `stock`, `open_orders`, `avail_now`
- `ordered_qty`, `shipped_qty`, `open_qty`, `avail_later`
- `priority`
- `latest_po_received_date`, `stock_age_range`

### Auto-Generated Features
- `order_fill_ratio` = `avail_now / open_orders`
- `total_inventory` = `stock + avail_now + avail_later`
- `stock_shortage` = `open_orders - avail_now`
- `has_backorder` = `1 if stock_shortage > 0 else 0`
- `days_since_po` = days since latest PO
- `product_size` = extracted from SKU
- `product_color` = extracted from SKU
- `stock_age_numeric` = numeric mapping of stock_age_range

### Encoded Features
All categorical features are automatically label-encoded.

---

## 🐳 Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t demand-predictor .
docker run -p 8000:8000 demand-predictor
```

---

## ☁️ Cloud Deployment Options

### AWS Lambda + API Gateway
1. Package model + code as Lambda function
2. Use Zappa or Serverless framework
3. Deploy with API Gateway

### Google Cloud Run
```bash
gcloud run deploy demand-predictor \
  --source . \
  --platform managed \
  --region us-central1
```

### Azure App Service
```bash
az webapp up \
  --name demand-predictor \
  --runtime "PYTHON:3.11"
```

### Heroku
```bash
heroku create demand-predictor
git push heroku main
```

---

## 📈 Monitoring & Logging

Add logging to track predictions:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Prediction request received")
    result = predictor.predict(data)
    logger.info(f"Prediction: {result['prediction']}")
    return jsonify(result)
```

---

## 🔒 Security Best Practices

1. **Add Authentication**
```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Implement your auth logic
    return True

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    ...
```

2. **Rate Limiting**
```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per hour"])

@app.route('/predict')
@limiter.limit("10 per minute")
def predict():
    ...
```

3. **Input Validation**
- Already handled by Pydantic in FastAPI
- Add custom validation for Flask

---

## 🔧 Troubleshooting

### Model Not Loading
- Ensure `best_model_fixed.pkl` is in the same directory
- Check Python version (3.8+)
- Verify all dependencies installed

### Prediction Errors
- Check input data format
- Ensure all required features present
- Check for missing/invalid values

### API Not Starting
- Check if port is already in use
- Verify firewall settings
- Check logs for detailed errors

---

## 📞 Support

For issues or questions:
1. Check the logs
2. Verify input data format
3. Review model requirements
4. Check API documentation at `/docs` (FastAPI)

---

## 📝 Change Log

### Version 1.0.0 (2025-11-29)
- Initial deployment
- Fixed data leakage issues
- Removed problematic features (fulfillment_rate, inventory_utilization, ordered_qty)
- Achieved realistic ~77% accuracy
- Deployed as Flask and FastAPI servers

---

## 🎓 Data Leakage Lessons Learned

This model went through important improvements to fix data leakage:

1. **First Leakage**: Features using `shipped_qty` to predict targets based on `shipped_qty`
   - Fixed by removing `fulfillment_rate` and `inventory_utilization`

2. **Second Leakage**: Using `ordered_qty` as feature when target = `(ordered_qty > threshold)`
   - Fixed by removing `ordered_qty` from features

**Result**: Realistic ~77% accuracy instead of misleading 100%

Lower accuracy with clean data is BETTER than perfect accuracy with data leakage!

---

## 📚 Additional Resources

- [Model Training Notebook](complete_eda_fixed.ipynb)
- [Data Leakage Explanation](complete_eda_fixed.ipynb#data-leakage)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built with ❤️ for Sports Merchandise Demand Forecasting**
