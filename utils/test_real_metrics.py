#!/usr/bin/env python3
"""
Real integration test: Sử dụng metrics.py với actual trained models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from app.ml.metrics import CryptoMetrics
from app.ml.data_prep import load_prepared_datasets
from app.ml.evaluate import load_trained_models
import joblib

def test_real_models_with_advanced_metrics():
    """Test advanced metrics với real trained models"""
    print("🎯 REAL MODEL EVALUATION WITH ADVANCED METRICS")
    print("=" * 80)
    
    try:
        # Load real data and models
        datasets = load_prepared_datasets('ml_datasets_top3')
        models = load_trained_models()
        
        test_df = datasets['test_df']
        feature_cols = datasets['feature_cols']
        
        print(f"📊 Test dataset: {len(test_df)} samples")
        print(f"🎯 Features: {len(feature_cols)} features")
        
        # Prepare test data
        X_test = test_df[feature_cols].fillna(0)
        y_test_price = test_df['target_price'].values
        
        print(f"💰 Price range: ${y_test_price.min():.2f} - ${y_test_price.max():.2f}")
        
        # Initialize advanced metrics
        evaluator = CryptoMetrics()
        
        print("\n" + "="*80)
        print("🚀 EVALUATING REAL TRAINED MODELS")
        print("="*80)
        
        model_results = {}
        
        # 1. Linear Regression Evaluation
        if 'linear_regression' in models:
            print("\n📈 LINEAR REGRESSION - Advanced Evaluation")
            print("-" * 60)
            
            lr_price = models['linear_regression']['price_model']
            lr_metadata = models['linear_regression']['metadata']
            
            # Get feature order from metadata or use default
            lr_feature_order = lr_metadata.get('feature_cols', feature_cols)
            X_test_lr = X_test[lr_feature_order]
            
            # Make predictions
            y_pred_lr = lr_price.predict(X_test_lr.values)
            
            # Advanced evaluation
            lr_metrics = evaluator.evaluate_regression(y_test_price, y_pred_lr, "LinearRegression")
            model_results['LinearRegression'] = lr_metrics
            
            print(f"  🎯 Advanced Metrics:")
            print(f"     MAE: ${lr_metrics['mae']:.2f}")
            print(f"     R²: {lr_metrics['r2']:.6f}")
            print(f"     MAPE: {lr_metrics['mape']:.2f}%")
            print(f"     Directional Accuracy: {lr_metrics['directional_accuracy']:.4f}")
            print(f"     Price Correlation: {lr_metrics['price_correlation']:.6f}")
            
            # Trading performance
            trading_lr = evaluator.evaluate_trading_performance(y_test_price, y_pred_lr)
            print(f"  💼 Trading Performance:")
            print(f"     Win Rate: {trading_lr['win_rate']:.2%}")
            print(f"     Sharpe Ratio: {trading_lr['sharpe_ratio']:.4f}")
            print(f"     Max Drawdown: {trading_lr['max_drawdown']:.2%}")
        
        # 2. KNN Evaluation  
        if 'knn' in models:
            print("\n🎯 KNN REGRESSOR - Advanced Evaluation")
            print("-" * 60)
            
            knn_regressor = models['knn']['regressor']['price_model']
            
            # Make predictions
            y_pred_knn = knn_regressor.predict(X_test.values)
            
            # Advanced evaluation
            knn_metrics = evaluator.evaluate_regression(y_test_price, y_pred_knn, "KNN")
            model_results['KNN'] = knn_metrics
            
            print(f"  🎯 Advanced Metrics:")
            print(f"     MAE: ${knn_metrics['mae']:.2f}")
            print(f"     R²: {knn_metrics['r2']:.6f}")
            print(f"     MAPE: {knn_metrics['mape']:.2f}%")
            print(f"     Directional Accuracy: {knn_metrics['directional_accuracy']:.4f}")
            print(f"     Price Correlation: {knn_metrics['price_correlation']:.6f}")
            
            # Trading performance
            trading_knn = evaluator.evaluate_trading_performance(y_test_price, y_pred_knn)
            print(f"  💼 Trading Performance:")
            print(f"     Win Rate: {trading_knn['win_rate']:.2%}")
            print(f"     Sharpe Ratio: {trading_knn['sharpe_ratio']:.4f}")
            print(f"     Max Drawdown: {trading_knn['max_drawdown']:.2%}")
            
        # 3. Classification Evaluation
        if 'knn' in models and 'classifier' in models['knn']:
            print("\n🎯 KNN CLASSIFIER - Advanced Evaluation")
            print("-" * 60)
            
            knn_clf = models['knn']['classifier']
            scaler = knn_clf['scaler']
            trend_model = knn_clf['trend_model']
            
            # Prepare data
            X_test_scaled = scaler.transform(X_test.values)
            y_test_trend = (test_df['returns'] > 0).astype(int)
            
            # Make predictions
            y_pred_trend = trend_model.predict(X_test_scaled)
            
            # Advanced classification evaluation
            clf_metrics = evaluator.evaluate_classification(
                y_test_trend, y_pred_trend, "KNN", "trend"
            )
            
            print(f"  🎯 Classification Metrics:")
            print(f"     Accuracy: {clf_metrics['accuracy']:.4f}")
            print(f"     Precision: {clf_metrics['precision']:.4f}")
            print(f"     F1-Score: {clf_metrics['f1']:.4f}")
            print(f"     Trend Precision: {clf_metrics['trend_precision']:.4f}")
            print(f"     Bull/Bear Ratio: {clf_metrics['bull_bear_ratio']:.2f}")
        
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON - REAL MODELS")
        print("="*80)
        
        # Compare models
        if model_results:
            comparison_df = evaluator.compare_models(model_results)
            print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE REAL MODEL REPORT")
        print("="*80)
        
        # Generate comprehensive report
        full_report = evaluator.generate_performance_report()
        print(full_report)
        
        print("\n" + "="*80)
        print("✅ REAL MODEL INTEGRATION TEST COMPLETE!")
        print("🎯 Advanced metrics successfully applied to production models")
        print("📊 Crypto-specific insights generated")
        print("💼 Trading performance quantified")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error in real model evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_models_with_advanced_metrics()