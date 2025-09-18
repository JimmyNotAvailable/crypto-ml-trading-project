#!/usr/bin/env python3
"""
🌳 RANDOM FOREST PERFORMANCE TEST
===============================

Test Random Forest performance và so sánh với các thuật toán khác
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.ml.smart_pipeline import SmartTrainingPipeline

def test_random_forest_performance():
    """Test Random Forest performance vs other algorithms"""
    
    print("🌳 RANDOM FOREST PERFORMANCE TEST")
    print("=" * 50)
    
    pipeline = SmartTrainingPipeline()
    
    # Test Random Forest regression cho price
    print("\n1️⃣ TESTING RANDOM FOREST REGRESSION (PRICE):")
    print("-" * 45)
    
    rf_result = pipeline.train_with_best_algorithm(
        target_type='price',
        force_algorithm='random_forest'
    )
    
    if rf_result['success']:
        rf_metrics = rf_result['training_result']['metrics']
        print(f"\n🌳 RANDOM FOREST KẾT QUẢ:")
        print(f"   📈 R² Score: {rf_metrics.get('r2_score', 0):.4f}")
        print(f"   📊 RMSE: {rf_metrics.get('rmse', 0):.4f}")
        print(f"   ⏱️ Training Time: {rf_metrics.get('training_time', 0):.2f}s")
        print(f"   🌲 Trees: {rf_metrics.get('n_estimators', 0)}")
    else:
        print(f"❌ Random Forest failed: {rf_result['error']}")
    
    # So sánh với Linear Regression
    print("\n2️⃣ COMPARING WITH LINEAR REGRESSION:")
    print("-" * 40)
    
    lr_result = pipeline.train_with_best_algorithm(
        target_type='price',
        force_algorithm='linear_regression'
    )
    
    if lr_result['success']:
        lr_metrics = lr_result['training_result']['metrics']
        print(f"\n📊 LINEAR REGRESSION KẾT QUẢ:")
        print(f"   📈 R² Score: {lr_metrics.get('r2_score', 0):.4f}")
        print(f"   📊 RMSE: {lr_metrics.get('rmse', 0):.4f}")
        print(f"   ⏱️ Training Time: {lr_metrics.get('training_time', 0):.2f}s")
    
    # So sánh kết quả
    if rf_result['success'] and lr_result['success']:
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"┌─────────────────────────────────────────┐")
        print(f"│ 🏁 ALGORITHM COMPARISON                 │")
        print(f"├─────────────────────────────────────────┤")
        
        rf_r2 = rf_metrics.get('r2_score', 0)
        lr_r2 = lr_metrics.get('r2_score', 0)
        rf_time = rf_metrics.get('training_time', 0)
        lr_time = lr_metrics.get('training_time', 0)
        
        print(f"│ Random Forest R²: {rf_r2:.4f}           │")
        print(f"│ Linear Reg R²:    {lr_r2:.4f}           │")
        print(f"│ RF Training Time: {rf_time:.2f}s             │")
        print(f"│ LR Training Time: {lr_time:.2f}s             │")
        print(f"├─────────────────────────────────────────┤")
        
        # Winner analysis
        if rf_r2 > lr_r2:
            accuracy_winner = "🌳 Random Forest"
            accuracy_diff = rf_r2 - lr_r2
        else:
            accuracy_winner = "📊 Linear Regression"
            accuracy_diff = lr_r2 - rf_r2
        
        if rf_time < lr_time:
            speed_winner = "🌳 Random Forest"
            speed_diff = lr_time - rf_time
        else:
            speed_winner = "📊 Linear Regression"
            speed_diff = rf_time - lr_time
        
        print(f"│ 🏆 Accuracy Winner: {accuracy_winner:15}  │")
        print(f"│    Difference: +{accuracy_diff:.4f}             │")
        print(f"│ ⚡ Speed Winner: {speed_winner:18}     │")
        print(f"│    Difference: -{speed_diff:.2f}s              │")
        print(f"└─────────────────────────────────────────┘")
    
    # Test classification cho price_change
    print(f"\n3️⃣ TESTING RANDOM FOREST CLASSIFICATION:")
    print("-" * 45)
    
    rf_class_result = pipeline.train_with_best_algorithm(
        target_type='price_change',
        force_algorithm='random_forest_classification'
    )
    
    if rf_class_result['success']:
        rf_class_metrics = rf_class_result['training_result']['metrics']
        print(f"\n🌳 RANDOM FOREST CLASSIFICATION KẾT QUẢ:")
        print(f"   🎯 Accuracy: {rf_class_metrics.get('accuracy', 0):.4f}")
        print(f"   📊 F1 Macro: {rf_class_metrics.get('f1_macro', 0):.4f}")
        print(f"   ⏱️ Training Time: {rf_class_metrics.get('training_time', 0):.2f}s")
    else:
        print(f"❌ Random Forest Classification failed: {rf_class_result['error']}")
    
    print(f"\n🎉 RANDOM FOREST TEST HOÀN TẤT!")

def demo_feature_importance():
    """Demo tính năng feature importance của Random Forest"""
    
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 40)
    
    # Load a trained Random Forest model and show feature importance
    # This would be implemented after we have a trained model
    print("📊 Feature importance analysis sẽ được hiển thị sau khi train model...")
    print("🌳 Random Forest có khả năng tự động xếp hạng tầm quan trọng của features")
    print("📈 Điều này rất hữu ích cho crypto prediction để biết features nào ảnh hưởng nhất")

if __name__ == "__main__":
    test_random_forest_performance()
    demo_feature_importance()