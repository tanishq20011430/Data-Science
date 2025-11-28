"""
Simple Example: Using the Deployed Model
=========================================

This script shows the simplest way to use the trained model.
"""

from deploy_model import DemandPredictor
import pandas as pd

# Load the model
print("Loading model...")
predictor = DemandPredictor('best_model_fixed.pkl')

print("\n" + "="*80)
print("EXAMPLE 1: Predict Single Product")
print("="*80)

# Example product
product = {
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

result = predictor.predict(product)

print(f"\nProduct: {product['fin_sku']}")
print(f"Team: {product['fin_team_name']}")
print(f"Stock: {product['stock']}")
print(f"Open Orders: {product['open_orders']}")
print(f"\n🎯 PREDICTION: {result['prediction']}")

if 'probability_high_demand' in result:
    print(f"\nProbabilities:")
    print(f"  High Demand: {result['probability_high_demand']:.1%}")
    print(f"  Low Demand:  {result['probability_low_demand']:.1%}")
    print(f"  Confidence:  {result['confidence']:.1%}")


print("\n" + "="*80)
print("EXAMPLE 2: Predict Multiple Products")
print("="*80)

# Multiple products
products = [
    {
        'country': 'Canada',
        'fin_sku': 'BTR6515792-BLK-L',
        'fin_brand': 'Pro Standard',
        'fin_team_name': 'TORONTO RAPTORS',
        'stock': 100.0,
        'open_orders': 50.0,
        'avail_now': 75.0,
        'avail_later': 20.0,
        'ordered_qty': 10.0,
        'shipped_qty': 8.0,
        'open_qty': 2.0,
        'fpo': 5.0,
        'priority': 1
    },
    {
        'country': 'USA',
        'fin_sku': 'HWJ567470-MDN-M',
        'fin_brand': 'Pro Standard',
        'fin_team_name': 'WINNIPEG JETS',
        'stock': 25.0,
        'open_orders': 30.0,
        'avail_now': 10.0,
        'avail_later': 5.0,
        'ordered_qty': 2.0,
        'shipped_qty': 1.0,
        'open_qty': 1.0,
        'fpo': 2.0,
        'priority': 2
    },
    {
        'country': 'Canada',
        'fin_sku': 'NBA123456-RED-XL',
        'fin_brand': 'Nike',
        'fin_team_name': 'LOS ANGELES LAKERS',
        'stock': 200.0,
        'open_orders': 100.0,
        'avail_now': 150.0,
        'avail_later': 50.0,
        'ordered_qty': 25.0,
        'shipped_qty': 20.0,
        'open_qty': 5.0,
        'fpo': 10.0,
        'priority': 1
    }
]

results = predictor.predict(pd.DataFrame(products))

print(f"\nPredictions for {len(results)} products:\n")
for i, (prod, res) in enumerate(zip(products, results), 1):
    emoji = "🔥" if res['prediction'] == "High Order Demand" else "📊"
    conf = res.get('confidence', 0)
    print(f"{emoji} {i}. {prod['fin_sku'][:20]:<20} → {res['prediction']:<20} (confidence: {conf:.1%})")


print("\n" + "="*80)
print("EXAMPLE 3: Batch Prediction from CSV")
print("="*80)

try:
    # Try to predict from matched_data.csv
    print("\nReading first 5 rows from matched_data.csv...")
    df = pd.read_csv('matched_data.csv').head(5)
    
    results = predictor.predict(df)
    
    print(f"\nPredictions:")
    for i, res in enumerate(results, 1):
        emoji = "🔥" if res['prediction'] == "High Order Demand" else "📊"
        print(f"{emoji} Row {i}: {res['prediction']}")
    
    print("\n✓ Batch prediction successful!")
    
except FileNotFoundError:
    print("\nmatched_data.csv not found. Skipping this example.")


print("\n" + "="*80)
print("✅ ALL EXAMPLES COMPLETED!")
print("="*80)

print("\nNext steps:")
print("1. Deploy as Flask API:   python flask_api.py")
print("2. Deploy as FastAPI:     uvicorn fastapi_api:app --reload")
print("3. Test the API:          python test_api.py")
print("\nSee DEPLOYMENT_GUIDE.md for complete documentation.")
