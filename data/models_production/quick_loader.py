
# Quick Model Loader for Discord Bot
import pickle
import numpy as np
import pandas as pd

def load_production_models():
    """Load production-ready crypto prediction models"""
    with open('data/models_production\crypto_models_production.pkl', 'rb') as f:
        return pickle.load(f)

def predict_price(features_dict):
    """Quick price prediction function"""
    models_package = load_production_models()
    
    # Convert features
    feature_cols = models_package['feature_cols']
    features_df = pd.DataFrame([features_dict])[feature_cols]
    
    # Predict
    price_pred = models_package['models']['price_predictor'].predict(features_df)[0]
    trend_pred = models_package['models']['trend_classifier'].predict(features_df)[0]
    
    return {
        'predicted_price': float(price_pred),
        'trend': int(trend_pred),
        'confidence': 'high' if models_package['performance']['price_predictor_r2'] > 0.8 else 'medium'
    }
