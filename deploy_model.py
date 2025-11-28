"""
Model Deployment Script - Sports Merchandise Demand Prediction
================================================================

This script provides a simple interface to load and use the trained model
for making predictions on new data.

Usage:
    python deploy_model.py
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DemandPredictor:
    """
    A class to load and use the trained demand prediction model.
    """
    
    def __init__(self, model_path='best_model_fixed.pkl'):
        """
        Initialize the predictor by loading the saved model package.
        
        Args:
            model_path (str): Path to the pickle file containing the model
        """
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            self.model_package = pickle.load(f)
        
        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.feature_names = self.model_package['feature_names']
        self.label_encoders = self.model_package['label_encoders']
        self.model_name = self.model_package['model_name']
        self.metrics = self.model_package['metrics']
        
        print(f"✓ Model loaded successfully!")
        print(f"  Model: {self.model_name}")
        print(f"  Accuracy: {self.metrics['Accuracy']:.4f}")
        print(f"  F1-Score: {self.metrics['F1-Score']:.4f}")
        print(f"  Features: {len(self.feature_names)}")
        
    def preprocess_data(self, data):
        """
        Preprocess input data to match the format expected by the model.
        
        Args:
            data (dict or pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        df = data.copy()
        
        # Handle missing values
        numerical_fill_cols = ['fpo', 'stock', 'open_orders', 'avail_now', 'ordered_qty', 
                               'shipped_qty', 'open_qty', 'avail_later']
        for col in numerical_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        categorical_fill_cols = ['pf_type', 'BOL', 'pick_created', 'pick_issued', 
                                 'pick_packed', 'pick_transit']
        for col in categorical_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Convert date columns to datetime
        date_cols = ['latest_po_received_date', 'pick_created_date', 'pick_issued_date', 
                     'pi_packed_date', 'pick_transit_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create engineered features
        # 1. Inventory efficiency metrics
        df['order_fill_ratio'] = np.where(df['open_orders'] > 0,
                                          df['avail_now'] / df['open_orders'],
                                          0)
        
        # 2. Stock status features
        df['total_inventory'] = df['stock'] + df['avail_now'] + df['avail_later']
        df['stock_shortage'] = df['open_orders'] - df['avail_now']
        df['has_backorder'] = (df['stock_shortage'] > 0).astype(int)
        
        # 3. Date-based features
        if 'latest_po_received_date' in df.columns:
            df['days_since_po'] = (pd.Timestamp.now() - df['latest_po_received_date']).dt.days
        else:
            df['days_since_po'] = 0
        
        # 4. Extract size from SKU
        if 'fin_sku' in df.columns:
            df['product_size'] = df['fin_sku'].apply(self._extract_size)
            df['product_color'] = df['fin_sku'].apply(self._extract_color)
        
        # 5. Stock age category
        stock_age_mapping = {
            '(1) 0-2m': 1,
            '(2) 4-6m': 2,
            '(3) 6-10m': 3,
            '(4) 10-12m': 4,
            '(5) 12-24m': 5,
            '(6) 24-36m': 6,
            '(7) 36m+': 7
        }
        if 'stock_age_range' in df.columns:
            df['stock_age_numeric'] = df['stock_age_range'].map(stock_age_mapping).fillna(0)
        
        # Encode categorical variables
        categorical_encode_cols = ['country', 'entity', 'currency', 'fin_brand', 'fin_team_name', 
                                   'fin_department', 'fin_gender', 'fin_class', 'fin_league_license',
                                   'location', 'pf_type', 'product_size', 'product_color']
        
        for col in categorical_encode_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unknown categories
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )
        
        # Select only the features used by the model
        # Add missing features with default values
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        X = df[self.feature_names].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        return X
    
    def _extract_size(self, sku):
        """Extract size from SKU"""
        if pd.isna(sku):
            return 'Unknown'
        parts = str(sku).split('-')
        if len(parts) >= 3:
            return parts[-1]
        return 'Unknown'
    
    def _extract_color(self, sku):
        """Extract color from SKU"""
        if pd.isna(sku):
            return 'Unknown'
        parts = str(sku).split('-')
        if len(parts) >= 2:
            return parts[-2]
        return 'Unknown'
    
    def predict(self, data):
        """
        Make predictions on new data.
        
        Args:
            data (dict or pd.DataFrame): Input data
            
        Returns:
            dict: Predictions with probabilities
        """
        # Preprocess data
        X = self.preprocess_data(data)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = None
        
        # Format results
        results = []
        for i in range(len(predictions)):
            result = {
                'prediction': 'High Order Demand' if predictions[i] == 1 else 'Low Order Demand',
                'prediction_value': int(predictions[i])
            }
            
            if probabilities is not None:
                result['probability_low_demand'] = float(probabilities[i][0])
                result['probability_high_demand'] = float(probabilities[i][1])
                result['confidence'] = float(max(probabilities[i]))
            
            results.append(result)
        
        return results if len(results) > 1 else results[0]
    
    def predict_batch(self, csv_path, output_path=None):
        """
        Make predictions on a CSV file.
        
        Args:
            csv_path (str): Path to input CSV file
            output_path (str): Path to save predictions (optional)
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"Making predictions on {len(df)} rows...")
        predictions = self.predict(df)
        
        # Add predictions to dataframe
        if isinstance(predictions, list):
            df['prediction'] = [p['prediction'] for p in predictions]
            df['prediction_value'] = [p['prediction_value'] for p in predictions]
            if 'probability_high_demand' in predictions[0]:
                df['probability_low_demand'] = [p['probability_low_demand'] for p in predictions]
                df['probability_high_demand'] = [p['probability_high_demand'] for p in predictions]
                df['confidence'] = [p['confidence'] for p in predictions]
        
        # Save to file if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✓ Predictions saved to {output_path}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DemandPredictor('best_model_fixed.pkl')
    
    # Example 1: Single prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Product Prediction")
    print("="*80)
    
    sample_product = {
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
    
    result = predictor.predict(sample_product)
    print(f"\nPrediction: {result['prediction']}")
    if 'probability_high_demand' in result:
        print(f"Probability of High Demand: {result['probability_high_demand']:.2%}")
        print(f"Probability of Low Demand: {result['probability_low_demand']:.2%}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    # Example 2: Batch prediction (if matched_data.csv exists)
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)
    
    try:
        # Read a sample of the data
        df_sample = pd.read_csv('matched_data.csv').head(10)
        predictions = predictor.predict(df_sample)
        
        print(f"\nPredictions for {len(predictions)} products:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['prediction']} (confidence: {pred.get('confidence', 'N/A')})")
        
        print("\n✓ Batch prediction successful!")
        
    except FileNotFoundError:
        print("\nmatched_data.csv not found. Skipping batch prediction example.")
    
    print("\n" + "="*80)
    print("Deployment script ready!")
    print("="*80)
    print("\nTo use this model in your application:")
    print("1. from deploy_model import DemandPredictor")
    print("2. predictor = DemandPredictor('best_model_fixed.pkl')")
    print("3. result = predictor.predict(your_data)")
