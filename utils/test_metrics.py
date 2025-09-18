#!/usr/bin/env python3
"""
Demo script để showcase giá trị của metrics.py
Sử dụng kết quả từ models đã train để demo advanced metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from app.ml.metrics import CryptoMetrics, quick_regression_eval, quick_trading_eval
from app.ml.data_prep import load_prepared_datasets
import joblib

def demo_advanced_metrics():
    """Demo comprehensive metrics evaluation"""
    print("🎯 DEMO: ADVANCED CRYPTO METRICS EVALUATION")
    print("=" * 80)
    
    # Load existing test data and model predictions
    try:
        datasets = load_prepared_datasets('ml_datasets_top3')
        test_df = datasets['test_df']
        
        # Create sample predictions for demo
        y_true_price = test_df['target_price'].values[:1000]  # Limit for demo
        
        # Simulate different model predictions with realistic performance
        np.random.seed(42)
        noise_levels = {
            'excellent_model': 0.01,    # Very good model
            'good_model': 0.05,         # Good model  
            'average_model': 0.15       # Average model
        }
        
        model_predictions = {}
        for model_name, noise_level in noise_levels.items():
            # Add realistic noise to true values
            noise = np.random.normal(0, noise_level * np.std(y_true_price), len(y_true_price))
            model_predictions[model_name] = y_true_price + noise
        
        print(f"📊 Testing with {len(y_true_price)} price predictions")
        print(f"💰 Price range: ${y_true_price.min():.2f} - ${y_true_price.max():.2f}")
        
    except Exception as e:
        print(f"⚠️  Using synthetic data for demo: {e}")
        # Generate synthetic data for demo
        np.random.seed(42)
        y_true_price = 30000 + np.cumsum(np.random.normal(0, 100, 1000))  # BTC-like price
        
        model_predictions = {
            'excellent_model': y_true_price + np.random.normal(0, 50, 1000),
            'good_model': y_true_price + np.random.normal(0, 200, 1000),
            'average_model': y_true_price + np.random.normal(0, 500, 1000)
        }
        
        print(f"📊 Using synthetic crypto price data: {len(y_true_price)} points")
        print(f"💰 Price range: ${y_true_price.min():.2f} - ${y_true_price.max():.2f}")
    
    # Initialize advanced metrics evaluator
    evaluator = CryptoMetrics()
    
    print("\n" + "="*60)
    print("📈 COMPREHENSIVE REGRESSION EVALUATION")
    print("="*60)
    
    regression_results = {}
    for model_name, y_pred in model_predictions.items():
        print(f"\n🎯 Evaluating {model_name.upper()}")
        print("-" * 40)
        
        # Use advanced metrics
        metrics = evaluator.evaluate_regression(y_true_price, y_pred, model_name)
        regression_results[model_name] = metrics
        
        # Display key metrics
        print(f"  📊 Traditional Metrics:")
        print(f"     MAE: ${metrics['mae']:.2f}")
        print(f"     RMSE: ${metrics['rmse']:.2f}")
        print(f"     R²: {metrics['r2']:.4f}")
        
        print(f"  🎯 Crypto-Specific Metrics:")
        print(f"     MAPE: {metrics['mape']:.2f}%")
        print(f"     SMAPE: {metrics['smape']:.2f}%")
        print(f"     Directional Accuracy: {metrics['directional_accuracy']:.4f}")
        print(f"     Price Correlation: {metrics['price_correlation']:.4f}")
        print(f"     Max Error: ${metrics['max_error']:.2f}")
    
    print("\n" + "="*60)
    print("💼 TRADING PERFORMANCE EVALUATION")
    print("="*60)
    
    for model_name, y_pred in model_predictions.items():
        print(f"\n🚀 Trading Performance: {model_name.upper()}")
        print("-" * 45)
        
        trading_metrics = evaluator.evaluate_trading_performance(
            y_true_price, y_pred, initial_investment=10000
        )
        
        print(f"  💰 Financial Results:")
        print(f"     Total Return: {trading_metrics['total_return']:.4f}")
        print(f"     ROI: {trading_metrics['roi']:.2%}")
        print(f"     Final Portfolio: ${trading_metrics['final_portfolio_value']:.2f}")
        
        print(f"  📊 Risk Metrics:")
        print(f"     Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
        print(f"     Max Drawdown: {trading_metrics['max_drawdown']:.2%}")
        print(f"     Win Rate: {trading_metrics['win_rate']:.2%}")
        print(f"     Profit Factor: {trading_metrics['profit_factor']:.4f}")
    
    print("\n" + "="*60)
    print("📋 MODEL COMPARISON TABLE")
    print("="*60)
    
    # Create comparison table
    comparison_df = evaluator.compare_models(regression_results)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    print("\n" + "="*60)
    print("📊 CLASSIFICATION DEMO")
    print("="*60)
    
    # Demo classification metrics
    np.random.seed(42)
    y_true_trend = np.random.choice([0, 1], 500, p=[0.45, 0.55])  # Slightly bullish market
    
    # Simulate different classifier performance
    classifiers = {
        'excellent_classifier': 0.85,  # 85% accuracy
        'good_classifier': 0.75,       # 75% accuracy
        'poor_classifier': 0.55        # 55% accuracy
    }
    
    for clf_name, accuracy in classifiers.items():
        # Generate predictions with target accuracy
        correct_preds = int(len(y_true_trend) * accuracy)
        y_pred_trend = y_true_trend.copy()
        
        # Flip some predictions to achieve target accuracy
        wrong_indices = np.random.choice(len(y_true_trend), 
                                       len(y_true_trend) - correct_preds, 
                                       replace=False)
        y_pred_trend[wrong_indices] = 1 - y_pred_trend[wrong_indices]
        
        print(f"\n🎯 {clf_name.upper()} - Trend Classification")
        print("-" * 50)
        
        clf_metrics = evaluator.evaluate_classification(
            y_true_trend, y_pred_trend, clf_name, "trend"
        )
        
        print(f"  📊 Performance:")
        print(f"     Accuracy: {clf_metrics['accuracy']:.4f}")
        print(f"     Precision: {clf_metrics['precision']:.4f}")
        print(f"     F1-Score: {clf_metrics['f1']:.4f}")
        print(f"     Trend Precision: {clf_metrics['trend_precision']:.4f}")
        print(f"     Bull/Bear Ratio: {clf_metrics['bull_bear_ratio']:.2f}")
    
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE PERFORMANCE REPORT")
    print("="*80)
    
    # Generate final report
    full_report = evaluator.generate_performance_report()
    print(full_report)
    
    print("\n" + "="*80)
    print("✅ METRICS DEMO COMPLETE!")
    print("🎯 Value of metrics.py demonstrated:")
    print("   • Advanced crypto-specific metrics")
    print("   • Trading performance evaluation") 
    print("   • Model comparison capabilities")
    print("   • Comprehensive reporting")
    print("   • Financial risk assessment")
    print("="*80)

if __name__ == "__main__":
    demo_advanced_metrics()