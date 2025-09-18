#!/usr/bin/env python3
"""
Model Retraining Pipeline with Cleaned Data
- Retrain all ML models with robust cleaned datasets
- Evaluate improvements in performance
- Save new models for production use
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

class ModelRetrainer:
    """Retrain all models with cleaned data"""
    
    def __init__(self, cleaned_datasets):
        self.datasets = cleaned_datasets
        self.models = {}
        self.performance = {}
        
    def train_regression_models(self):
        """Train regression models for price prediction"""
        print("🎯 TRAINING REGRESSION MODELS")
        print("=" * 60)
        
        X_train = self.datasets['X_train']
        y_train = self.datasets['y_train']['price']
        X_test = self.datasets['X_test']
        y_test = self.datasets['y_test']['price']
        
        # Models to train
        regression_models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        for name, model in regression_models.items():
            print(f"📊 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            self.models[name] = model
            self.performance[name] = {
                'type': 'regression',
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'feature_importance': getattr(model, 'feature_importances_', None)
            }
            
            print(f"  ✅ {name}:")
            print(f"     Train MAE: ${train_mae:.2f} | R²: {train_r2:.3f}")
            print(f"     Test MAE:  ${test_mae:.2f} | R²: {test_r2:.3f}")
    
    def train_clustering_models(self):
        """Train clustering models"""
        print(f"\n🎯 TRAINING CLUSTERING MODELS")
        print("=" * 60)
        
        X_train = self.datasets['X_train']
        
        # K-Means clustering
        print("📊 Training K-Means clustering...")
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_train)
        
        # Evaluate clustering quality
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_train, cluster_labels)
        
        self.models['kmeans'] = kmeans
        self.performance['kmeans'] = {
            'type': 'clustering',
            'silhouette_score': silhouette_avg,
            'n_clusters': 5,
            'inertia': kmeans.inertia_
        }
        
        print(f"  ✅ K-Means:")
        print(f"     Silhouette Score: {silhouette_avg:.3f}")
        print(f"     Inertia: {kmeans.inertia_:.2f}")
    
    def train_knn_models(self):
        """Train KNN models for regression and classification"""
        print(f"\n🎯 TRAINING KNN MODELS")
        print("=" * 60)
        
        X_train = self.datasets['X_train']
        X_test = self.datasets['X_test']
        
        # KNN Regression for price prediction
        print("📊 Training KNN Regression...")
        
        y_train_price = self.datasets['y_train']['price']
        y_test_price = self.datasets['y_test']['price']
        
        knn_regressor = KNeighborsRegressor(n_neighbors=5)
        knn_regressor.fit(X_train, y_train_price)
        
        y_train_pred = knn_regressor.predict(X_train)
        y_test_pred = knn_regressor.predict(X_test)
        
        train_mae = mean_absolute_error(y_train_price, y_train_pred)
        test_mae = mean_absolute_error(y_test_price, y_test_pred)
        train_r2 = r2_score(y_train_price, y_train_pred)
        test_r2 = r2_score(y_test_price, y_test_pred)
        
        self.models['knn_regressor'] = knn_regressor
        self.performance['knn_regressor'] = {
            'type': 'regression',
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"  ✅ KNN Regression:")
        print(f"     Train MAE: ${train_mae:.2f} | R²: {train_r2:.3f}")
        print(f"     Test MAE:  ${test_mae:.2f} | R²: {test_r2:.3f}")
        
        # KNN Classification for trend prediction
        print("📊 Training KNN Classification...")
        
        y_train_trend = self.datasets['y_train']['trend']
        y_test_trend = self.datasets['y_test']['trend']
        
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train, y_train_trend)
        
        y_train_pred_class = knn_classifier.predict(X_train)
        y_test_pred_class = knn_classifier.predict(X_test)
        
        train_acc = accuracy_score(y_train_trend, y_train_pred_class)
        test_acc = accuracy_score(y_test_trend, y_test_pred_class)
        
        self.models['knn_classifier'] = knn_classifier
        self.performance['knn_classifier'] = {
            'type': 'classification',
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"  ✅ KNN Classification:")
        print(f"     Train Accuracy: {train_acc:.3f}")
        print(f"     Test Accuracy:  {test_acc:.3f}")
    
    def compare_with_original_models(self):
        """Compare new models with original ones"""
        print(f"\n📊 COMPARING WITH ORIGINAL MODELS")
        print("=" * 60)
        
        try:
            # Load original models
            from app.ml.evaluate import load_trained_models
            original_models = load_trained_models()
            
            # Load original datasets
            with open('data/cache/ml_datasets_top3.pkl', 'rb') as f:
                original_datasets = pickle.load(f)
            
            print("✅ Original models loaded for comparison")
            
            # Compare regression performance
            if 'linear_regression' in original_models and 'linear_regression' in self.models:
                # Test on original data
                X_test_orig = original_datasets['X_test']
                y_test_orig = original_datasets['y_test']['price']
                
                orig_pred = original_models['linear_regression'].predict(X_test_orig)
                orig_mae = mean_absolute_error(y_test_orig, orig_pred)
                orig_r2 = r2_score(y_test_orig, orig_pred)
                
                # New model performance
                new_mae = self.performance['linear_regression']['test_mae']
                new_r2 = self.performance['linear_regression']['test_r2']
                
                print(f"🔄 Linear Regression Comparison:")
                print(f"   Original: MAE=${orig_mae:.2f}, R²={orig_r2:.3f}")
                print(f"   New:      MAE=${new_mae:.2f}, R²={new_r2:.3f}")
                
                improvement_mae = ((orig_mae - new_mae) / orig_mae) * 100
                improvement_r2 = ((new_r2 - orig_r2) / abs(orig_r2)) * 100 if orig_r2 != 0 else 0
                
                print(f"   🎯 Improvement: MAE {improvement_mae:+.1f}%, R² {improvement_r2:+.1f}%")
                
        except Exception as e:
            print(f"⚠️ Could not load original models for comparison: {e}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance from trained models"""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        feature_names = self.datasets['feature_cols']
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_importance
            }).sort_values('importance', ascending=False)
            
            print("🌲 Random Forest Feature Importance:")
            for _, row in feature_importance_df.head(10).iterrows():
                print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    def save_retrained_models(self):
        """Save all retrained models"""
        print(f"\n💾 SAVING RETRAINED MODELS")
        print("=" * 60)
        
        # Create models directory
        models_dir = 'data/models_v2_clean'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Saved {name} to {model_path}")
        
        # Save performance metrics
        performance_path = os.path.join(models_dir, 'performance_metrics.pkl')
        with open(performance_path, 'wb') as f:
            pickle.dump(self.performance, f)
        print(f"✅ Saved performance metrics to {performance_path}")
        
        # Create combined models file for easy loading
        combined_models = {
            'models': self.models,
            'performance': self.performance,
            'feature_cols': self.datasets['feature_cols'],
            'scalers': {
                'robust_scaler': self.datasets['robust_scaler'],
                'power_transformer': self.datasets['power_transformer']
            },
            'metadata': {
                'training_date': pd.Timestamp.now(),
                'data_version': 'v2_clean',
                'cleaning_method': 'advanced_robust'
            }
        }
        
        combined_path = os.path.join(models_dir, 'retrained_models_v2.pkl')
        with open(combined_path, 'wb') as f:
            pickle.dump(combined_models, f)
        print(f"✅ Saved combined models to {combined_path}")
        
        return models_dir

def main():
    """Main model retraining pipeline"""
    print("🚀 MODEL RETRAINING PIPELINE WITH CLEANED DATA")
    print("=" * 80)
    
    # Load cleaned datasets
    print("📂 Loading cleaned datasets...")
    try:
        with open('data/cache/ml_datasets_top3_v2_clean.pkl', 'rb') as f:
            cleaned_datasets = pickle.load(f)
        print("✅ Cleaned datasets loaded successfully")
    except FileNotFoundError:
        print("❌ Error: Cleaned datasets not found! Run data_quality_fix_v2.py first.")
        return
    
    print(f"📊 Dataset info:")
    print(f"   Training samples: {len(cleaned_datasets['X_train'])}")
    print(f"   Test samples: {len(cleaned_datasets['X_test'])}")
    print(f"   Features: {len(cleaned_datasets['feature_cols'])}")
    
    # Initialize retrainer
    retrainer = ModelRetrainer(cleaned_datasets)
    
    # Train all models
    print(f"\n" + "="*80)
    print("🎯 MODEL TRAINING PHASE")
    print("="*80)
    
    retrainer.train_regression_models()
    retrainer.train_clustering_models()
    retrainer.train_knn_models()
    
    # Analysis
    print(f"\n" + "="*80)
    print("📊 ANALYSIS PHASE")
    print("="*80)
    
    retrainer.compare_with_original_models()
    retrainer.analyze_feature_importance()
    
    # Save models
    print(f"\n" + "="*80)
    print("💾 SAVING PHASE")
    print("="*80)
    
    models_dir = retrainer.save_retrained_models()
    
    # Final summary
    print(f"\n" + "="*80)
    print("🎯 RETRAINING COMPLETE")
    print("="*80)
    
    print("✅ All models retrained with cleaned data")
    print("✅ Performance metrics calculated")
    print("✅ Feature importance analyzed")
    print(f"✅ Models saved to: {models_dir}")
    
    print(f"\n📈 Retrained Models Summary:")
    for name, perf in retrainer.performance.items():
        if perf['type'] == 'regression':
            print(f"   {name}: MAE=${perf['test_mae']:.2f}, R²={perf['test_r2']:.3f}")
        elif perf['type'] == 'classification':
            print(f"   {name}: Accuracy={perf['test_accuracy']:.3f}")
        elif perf['type'] == 'clustering':
            print(f"   {name}: Silhouette={perf['silhouette_score']:.3f}")
    
    print(f"\n🚀 Ready for production deployment!")
    
    return retrainer

if __name__ == "__main__":
    main()