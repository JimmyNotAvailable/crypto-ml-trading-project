#!/usr/bin/env python3
"""Simple prediction test without app dependencies."""

import pickle
import pandas as pd
import sys
import os

# Add repo root to sys.path to resolve app imports
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

PROD_PATH = r"E:\Code on PC\DoAnMLPython\crypto-project-clean\data\models_production\crypto_models_production.pkl"

def predict_price_safe(features_dict):
    """Safe prediction that handles import issues."""
    try:
        with open(PROD_PATH, 'rb') as f:
            pkg = pickle.load(f)
        
        feats = pkg.get('feature_cols') or []
        df = pd.DataFrame([features_dict])
        if feats:
            df = df.reindex(columns=feats, fill_value=0)
        
        price_model = pkg['models']['price_predictor']
        trend_model = pkg['models']['trend_classifier']
        
        price = float(price_model.predict(df)[0])
        
        # Simple trend prediction
        try:
            t_pred = trend_model.predict(df)
            if hasattr(t_pred, '__getitem__'):
                trend = int(t_pred[0])
            else:
                trend = int(t_pred)
        except Exception:
            trend = 1
        
        conf = 'high' if (pkg.get('performance', {}).get('price_predictor_r2', 0) > 0.7) else 'medium'
        best_price = pkg.get('metadata', {}).get('best_price_model', '?')
        best_trend = pkg.get('metadata', {}).get('best_trend_model', '?')
        
        return {
            'predicted_price': price,
            'trend': trend,
            'confidence': conf,
            'model_name': f"{best_price}+{best_trend}",
            'metrics': {'r2': pkg.get('performance', {}).get('price_predictor_r2', None)},
        }
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None

if __name__ == '__main__':
    # Test
    features = {
        "open": 6000 * 0.998,
        "high": 6000 * 1.002,
        "low": 6000 * 0.995,
        "close": 6000,
        "volume": 50_000,
        "ma_10": 6000 * 0.996,
        "ma_50": 6000 * 0.97,
        "volatility": 2.5,
        "returns": 0.15,
        "hour": 12,
    }
    
    result = predict_price_safe(features)
    print("Prediction result:", result)