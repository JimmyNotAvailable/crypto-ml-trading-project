#!/usr/bin/env python3
"""
Train core models (Linear, Logistic, KNN, RandomForest) on prepared datasets
and export a production package with metrics + a quick_loader usable by the bot.
"""

import os
import sys
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path for 'app' imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.ml.data_prep import load_prepared_datasets
from app.ml.algorithms import (
    LinearRegressionModel,
    LogisticRegressionModel,
    KNNRegressor,
    KNNClassifier,
    RandomForestModel,
)


def root_dir():
    return os.path.dirname(os.path.abspath(os.path.join(__file__, "..", "..")))


def train_all(dataset_name: str = 'ml_datasets_top3'):
    datasets = load_prepared_datasets(dataset_name)
    # Build compatibility dataset for classes expecting 'train'/'test'
    datasets_compat = {
        'train': datasets.get('train_df', pd.DataFrame()),
        'test': datasets.get('test_df', pd.DataFrame())
    }

    results = {}

    # Linear Regression (price)
    lin = LinearRegressionModel(target_type='price', normalize_features=True)
    results['linear_regression'] = lin.train(datasets_compat)

    # Logistic Regression (trend)
    logit = LogisticRegressionModel()
    results['logistic_regression'] = logit.train(datasets_compat)

    # KNN Regressor (price)
    knnr = KNNRegressor(target_type='price', auto_tune=True)
    results['knn_regressor'] = knnr.train(datasets_compat)

    # KNN Classifier (trend)
    knnc = KNNClassifier(auto_tune=True)
    results['knn_classifier'] = knnc.train(datasets_compat)

    # Random Forest Regressor (price) — optional (skip by default using env ML_SKIP_RF=1)
    rf_reg = None
    skip_rf = os.getenv('ML_SKIP_RF', '1') == '1'
    if skip_rf:
        print("⏭️ Skipping RandomForest training (set ML_SKIP_RF=0 to enable).")
    else:
        try:
            rf_reg = RandomForestModel(task_type='regression', target_type='price', n_estimators=50, n_jobs=1, max_depth=12)
            results['rf_regression'] = rf_reg.train(datasets_compat)
        except Exception as e:
            print(f"⚠️ RandomForest training skipped due to error: {e}")

    # Choose best price predictor by test_r2 or train_r2 fallback
    candidates = [
        ('linear', lin),
        ('knn', knnr),
    ]
    if rf_reg is not None:
        candidates.append(('rf', rf_reg))
    def r2_of(m):
        metrics = (m.training_history or {}).get('metrics', {})
        return float(metrics.get('test_r2') or metrics.get('train_r2') or -1)
    best_price_name, best_price_model = max(candidates, key=lambda t: r2_of(t[1]))

    # Choose best trend classifier by test_accuracy or train_accuracy
    clf_candidates = [ ('logit', logit), ('knn', knnc) ]
    def acc_of(m):
        metrics = (m.training_history or {}).get('metrics', {})
        return float(metrics.get('test_accuracy') or metrics.get('train_accuracy') or -1)
    best_cls_name, best_cls_model = max(clf_candidates, key=lambda t: acc_of(t[1]))

    feature_cols = getattr(lin, 'feature_columns', None) or getattr(knnr, 'feature_columns', None)

    # Build production package
    package = {
        'models': {
            'price_predictor': best_price_model,
            'trend_classifier': best_cls_model,
        },
        'feature_cols': feature_cols or [],
        'performance': {
            'price_predictor_r2': r2_of(best_price_model),
            'trend_classifier_accuracy': acc_of(best_cls_model)
        },
        'metadata': {
            'created_date': datetime.now().isoformat(),
            'dataset': dataset_name,
            'best_price_model': best_price_name,
            'best_trend_model': best_cls_name,
        }
    }

    prod_dir = os.path.join(root_dir(), 'data', 'models_production')
    os.makedirs(prod_dir, exist_ok=True)

    prod_path = os.path.join(prod_dir, 'crypto_models_production.pkl')
    with open(prod_path, 'wb') as f:
        pickle.dump(package, f)

    # Write quick_loader
    ql_path = os.path.join(prod_dir, 'quick_loader.py')
    ql_code = """
import pickle
import pandas as pd

PROD_PATH = r"__PROD_PATH__"

def load_production_models():
    with open(PROD_PATH, 'rb') as f:
        return pickle.load(f)

def predict_price(features_dict):
    pkg = load_production_models()
    feats = pkg.get('feature_cols') or []
    df = pd.DataFrame([features_dict])
    if feats:
        df = df.reindex(columns=feats, fill_value=0)
    price_model = pkg['models']['price_predictor']
    trend_model = pkg['models']['trend_classifier']
    price = float(price_model.predict(df)[0])
    # Robust trend prediction handling different return types
    t_pred = getattr(trend_model, 'predict', lambda X: [1])(df)
    try:
        import pandas as _pd
        if isinstance(t_pred, _pd.DataFrame):
            t_pred = t_pred.iloc[:, 0].to_numpy()
    except Exception:
        pass
    try:
        trend = int(getattr(t_pred, '__getitem__', lambda i: t_pred)[0])
    except Exception:
        try:
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
""".replace("__PROD_PATH__", prod_path)
    with open(ql_path, 'w', encoding='utf-8') as f:
        f.write(ql_code)

    print(f"✅ Exported production package: {prod_path}")
    print(f"✅ Wrote quick_loader: {ql_path}")
    print(f"⭐ Best models: price={best_price_name} (R²={package['performance']['price_predictor_r2']:.3f}), trend={best_cls_name} (Acc={package['performance']['trend_classifier_accuracy']:.3f})")

    return prod_path


if __name__ == '__main__':
    train_all()
