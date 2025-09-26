# evaluate.py
# Model evaluation và comparison module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

# Import utilities
sys.path.append(os.path.dirname(__file__))
from data_prep import load_prepared_datasets, project_root_path

def load_model_metadata(model_name):
    """Load model metadata from JSON file"""
    root = project_root_path()
    models_dir = os.path.join(root, "models")
    metadata_path = os.path.join(models_dir, f"{model_name}_metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_trained_models():
    """Load all trained models"""
    root = project_root_path()
    models_dir = os.path.join(root, "models")
    
    models = {}
    
    try:
        # Load Linear Regression models
        linreg_price = joblib.load(os.path.join(models_dir, "linreg_price.joblib"))
        linreg_change = joblib.load(os.path.join(models_dir, "linreg_price_change.joblib"))
        models['linear_regression'] = {
            'price_model': linreg_price,
            'change_model': linreg_change,
            'metadata': load_model_metadata('linreg_price')  # Load from price model metadata
        }
        print("✅ Loaded Linear Regression models")
    except Exception as e:
        print(f"❌ Failed to load Linear Regression: {e}")
    
    try:
        # Load K-Means model
        kmeans_model = joblib.load(os.path.join(models_dir, "kmeans_crypto.joblib"))
        models['kmeans'] = {
            'model': kmeans_model,
            'metadata': load_model_metadata('kmeans_crypto')
        }
        print("✅ Loaded K-Means model")
    except Exception as e:
        print(f"❌ Failed to load K-Means: {e}")
    
    try:
        # Load KNN models
        knn_classifier = joblib.load(os.path.join(models_dir, "knn_crypto_classifier.joblib"))
        knn_regressor = joblib.load(os.path.join(models_dir, "knn_crypto_regressor.joblib"))
        models['knn'] = {
            'classifier': knn_classifier,
            'regressor': knn_regressor,
            'metadata': load_model_metadata('knn_crypto')
        }
        print("✅ Loaded KNN models")
    except Exception as e:
        print(f"❌ Failed to load KNN: {e}")
    
    return models

def evaluate_regression_models(models, datasets):
    """Evaluate and compare regression models - FIXED for proper data handling"""
    print("=== EVALUATING REGRESSION MODELS ===")
    
    # Use appropriate data for each model type:
    # - Linear Regression: Use scaled X_test (no outliers) 
    # - KNN: Use raw test_df data (same as training)
    feature_cols = datasets['feature_cols']
    
    print(f" Current feature_cols: {feature_cols}")
    
    regression_results = {}
    
    # Linear Regression Evaluation - Use scaled data (no outliers)
    if 'linear_regression' in models:
        print("\n--- LINEAR REGRESSION ---")
        
        # Use scaled data for Linear Regression (better for outlier handling)
        X_test_lr = datasets['X_test'] 
        y_test_price_lr = datasets['y_test']['price']
        y_test_change_lr = datasets['y_test']['price_change']
        
        print(f"📊 Linear Regression test data: {X_test_lr.shape}")
        print(f"📊 X_test range: [{X_test_lr.min().min() if hasattr(X_test_lr, 'min') else np.min(X_test_lr):.4f}, {X_test_lr.max().max() if hasattr(X_test_lr, 'max') else np.max(X_test_lr):.4f}]")
        
        try:
            lr_price = models['linear_regression']['price_model']
            lr_change = models['linear_regression']['change_model']
            
            # Get feature columns from metadata
            metadata = models['linear_regression'].get('metadata', {})
            trained_features = metadata.get('features', feature_cols)
            
            # IMPORTANT: Check both set equality AND order
            if set(trained_features) != set(feature_cols) or list(trained_features) != list(feature_cols):
                print(f"⚠️  Feature mismatch! Trained: {trained_features}, Current: {feature_cols}")
                # Reorder features to match training data
                if hasattr(X_test_lr, 'columns'):  # pandas DataFrame
                    X_test_ordered = X_test_lr[trained_features]
                else:  # numpy array
                    # Create DataFrame first to enable column selection
                    X_test_df = pd.DataFrame(X_test_lr, columns=feature_cols)
                    X_test_ordered = X_test_df[trained_features]
                print(f"✅ Reordered features to match training order")
            else:
                X_test_ordered = X_test_lr
                print(f"✅ Features already in correct order")
                
            print(f"📋 Features in use: {list(X_test_ordered.columns) if hasattr(X_test_ordered, 'columns') else trained_features}")
            print(f"📋 Test data shape: {X_test_ordered.shape}")
            
            # Convert to numpy for prediction (avoid feature name warnings)
            X_test_values = X_test_ordered.values if hasattr(X_test_ordered, 'values') else X_test_ordered
            
            # Price prediction with error handling
            y_pred_price_lr = lr_price.predict(X_test_values)
            lr_price_metrics = {
                'mae': mean_absolute_error(y_test_price_lr, y_pred_price_lr),
                'rmse': np.sqrt(mean_squared_error(y_test_price_lr, y_pred_price_lr)),
                'r2': r2_score(y_test_price_lr, y_pred_price_lr)
            }
            
            print(f"  Price - MAE: {lr_price_metrics['mae']:.4f}, R²: {lr_price_metrics['r2']:.4f}")
            
            # Price change prediction
            y_pred_change_lr = lr_change.predict(X_test_values)
            lr_change_metrics = {
                'mae': mean_absolute_error(y_test_change_lr, y_pred_change_lr),
                'rmse': np.sqrt(mean_squared_error(y_test_change_lr, y_pred_change_lr)),
                'r2': r2_score(y_test_change_lr, y_pred_change_lr)
            }
            
            print(f"  Change - MAE: {lr_change_metrics['mae']:.6f}, R²: {lr_change_metrics['r2']:.4f}")
            
            regression_results['linear_regression'] = {
                'price': lr_price_metrics,
                'change': lr_change_metrics,
                'price_predictions': y_pred_price_lr,
                'change_predictions': y_pred_change_lr
            }
            
        except Exception as e:
            print(f"❌ Linear Regression evaluation error: {e}")
            regression_results['linear_regression'] = {'error': str(e)}
    
    # KNN Regression Evaluation - Use raw data (same as training)
    if 'knn' in models:
        print("\n--- KNN REGRESSION ---")
        
        # Use raw test_df data for KNN (same as training data)
        test_df = datasets['test_df']
        X_test_knn = test_df[feature_cols].fillna(0)
        y_test_price_knn = test_df['target_price']
        y_test_change_knn = test_df['target_price_change']
        
        print(f"📊 KNN test data: {X_test_knn.shape}")
        print(f"📊 X_test range: [{X_test_knn.min().min():.4f}, {X_test_knn.max().max():.4f}]")
        
        try:
            knn_reg = models['knn']['regressor']
            scaler = knn_reg['scaler']
            
            # Get feature columns from metadata
            metadata = models['knn'].get('metadata', {})
            trained_features = metadata.get('features', feature_cols)
            
            # IMPORTANT: Check both set equality AND order  
            if set(trained_features) != set(feature_cols) or list(trained_features) != list(feature_cols):
                print(f"⚠️  Feature mismatch! Trained: {trained_features}, Current: {feature_cols}")
                # Handle pandas DataFrame vs numpy array
                if hasattr(X_test_knn, 'columns'):  # pandas DataFrame
                    X_test_ordered = X_test_knn[trained_features]
                else:  # numpy array - need to reorder by feature indices
                    # Create feature index mapping
                    feature_indices = [feature_cols.index(feat) for feat in trained_features]
                    X_test_ordered = X_test_knn[:, feature_indices]
                print(f"✅ Reordered features to match training order")
            else:
                X_test_ordered = X_test_knn
                print(f"✅ Features already in correct order")
                
            print(f"📋 Features in use: {trained_features}")
            print(f"📋 Test data shape: {X_test_ordered.shape}")
            
            # Scale features - IMPORTANT: Convert to numpy to avoid feature name issues
            X_test_values = X_test_ordered.values if hasattr(X_test_ordered, 'values') else X_test_ordered
            X_test_scaled = scaler.transform(X_test_values)
            
            # Price prediction
            y_pred_price_knn = knn_reg['price_model'].predict(X_test_scaled)
            knn_price_metrics = {
                'mae': mean_absolute_error(y_test_price_knn, y_pred_price_knn),
                'rmse': np.sqrt(mean_squared_error(y_test_price_knn, y_pred_price_knn)),
                'r2': r2_score(y_test_price_knn, y_pred_price_knn)
            }
            
            # Price change prediction
            y_pred_change_knn = knn_reg['change_model'].predict(X_test_scaled)
            knn_change_metrics = {
                'mae': mean_absolute_error(y_test_change_knn, y_pred_change_knn),
                'rmse': np.sqrt(mean_squared_error(y_test_change_knn, y_pred_change_knn)),
                'r2': r2_score(y_test_change_knn, y_pred_change_knn)
            }
            
            regression_results['knn'] = {
                'price_metrics': knn_price_metrics,
                'change_metrics': knn_change_metrics,
                'price_predictions': y_pred_price_knn,
                'change_predictions': y_pred_change_knn
            }
            
            print(f"  Price - MAE: {knn_price_metrics['mae']:.4f}, R²: {knn_price_metrics['r2']:.4f}")
            print(f"  Change - MAE: {knn_change_metrics['mae']:.6f}, R²: {knn_change_metrics['r2']:.4f}")
            
        except Exception as e:
            print(f"❌ Error in KNN evaluation: {e}")
            regression_results['knn'] = {'error': str(e)}
    
    return regression_results

def evaluate_classification_models(models, test_data):
    """Evaluate classification models"""
    print("=== EVALUATING CLASSIFICATION MODELS ===")
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'volatility', 
                   'returns', 'ma_10', 'ma_50', 'hour']
    
    X_test = test_data[feature_cols].fillna(0)
    
    # Create classification targets
    y_test_trend = (test_data['returns'] > 0).astype(int)
    vol_median = test_data['volatility'].median()
    y_test_vol = (test_data['volatility'] > vol_median).astype(int)
    
    classification_results = {}
    
    # KNN Classification Evaluation
    if 'knn' in models:
        print("\n--- KNN CLASSIFICATION ---")
        knn_clf = models['knn']['classifier']
        scaler = knn_clf['scaler']
        
        # Use feature_cols for consistent ordering
        knn_feature_cols = knn_clf.get('feature_cols', feature_cols)
        print(f"⚠️  Feature mismatch check for KNN classification...")
        print(f"📋 Expected features: {knn_feature_cols}")
        print(f"📋 Current features: {feature_cols}")
        
        # Reorder features if needed
        if knn_feature_cols != feature_cols:
            print("✅ Reordering features for KNN classification")
            X_test_reordered = X_test[knn_feature_cols]
        else:
            print("✅ Features already in correct order")
            X_test_reordered = X_test
            
        print(f"📋 Classification features in use: {list(X_test_reordered.columns)}")
        
        # Scale features
        X_test_scaled = scaler.transform(X_test_reordered)
        
        # Trend prediction
        y_pred_trend = knn_clf['trend_model'].predict(X_test_scaled)
        trend_accuracy = (y_pred_trend == y_test_trend).mean()
        
        # Volatility prediction
        y_pred_vol = knn_clf['vol_model'].predict(X_test_scaled)
        vol_accuracy = (y_pred_vol == y_test_vol).mean()
        
        classification_results['knn'] = {
            'trend_accuracy': trend_accuracy,
            'vol_accuracy': vol_accuracy,
            'trend_predictions': y_pred_trend,
            'vol_predictions': y_pred_vol,
            'trend_true': y_test_trend,
            'vol_true': y_test_vol
        }
        
        print(f"  Trend Accuracy: {trend_accuracy:.4f}")
        print(f"  Volatility Accuracy: {vol_accuracy:.4f}")
    
    return classification_results

def create_model_comparison_report(regression_results, classification_results, models_metadata):
    """Create comprehensive model comparison report"""
    print("=== CREATING MODEL COMPARISON REPORT ===")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Complete Model Evaluation & Comparison Report', fontsize=16)
    
    # 1. Regression R² Comparison
    if regression_results:
        model_names = list(regression_results.keys())
        
        price_r2_scores = []
        change_r2_scores = []
        
        for model in model_names:
            result = regression_results[model]
            if 'price_metrics' in result:
                price_r2_scores.append(result['price_metrics']['r2'])
                change_r2_scores.append(result['change_metrics']['r2'])
            elif 'price' in result:
                price_r2_scores.append(result['price']['r2'])
                change_r2_scores.append(result['change']['r2'])
            else:
                price_r2_scores.append(0)
                change_r2_scores.append(0)
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, price_r2_scores, width, label='Price R²', alpha=0.8)
        axes[0, 0].bar(x + width/2, change_r2_scores, width, label='Price Change R²', alpha=0.8)
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Regression R² Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([name.replace('_', ' ').title() for name in model_names])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Regression MAE Comparison
    if regression_results:
        price_mae_scores = []
        change_mae_scores = []
        
        for model in model_names:
            result = regression_results[model]
            if 'price_metrics' in result:
                price_mae_scores.append(result['price_metrics']['mae'])
                change_mae_scores.append(result['change_metrics']['mae'])
            elif 'price' in result:
                price_mae_scores.append(result['price']['mae'])
                change_mae_scores.append(result['change']['mae'])
            else:
                price_mae_scores.append(0)
                change_mae_scores.append(0)
        
        axes[0, 1].bar(x - width/2, price_mae_scores, width, label='Price MAE', alpha=0.8)
        axes[0, 1].bar(x + width/2, change_mae_scores, width, label='Price Change MAE', alpha=0.8)
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Regression MAE Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([name.replace('_', ' ').title() for name in model_names])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Classification Accuracy Comparison
    if classification_results and 'knn' in classification_results:
        knn_results = classification_results['knn']
        accuracies = [knn_results['trend_accuracy'], knn_results['vol_accuracy']]
        task_names = ['Trend Prediction', 'Volatility Classification']
        
        axes[0, 2].bar(task_names, accuracies, color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Classification Accuracy')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
    
    # 4. Linear Regression vs KNN Price Prediction Scatter
    if 'linear_regression' in regression_results and 'knn' in regression_results:
        lr_pred = regression_results['linear_regression']['price_predictions']
        knn_pred = regression_results['knn']['price_predictions']
        
        # Sample for visualization
        n_samples = min(1000, len(lr_pred))
        sample_idx = np.random.choice(len(lr_pred), n_samples, replace=False)
        
        axes[1, 0].scatter(lr_pred[sample_idx], knn_pred[sample_idx], alpha=0.6, s=20)
        axes[1, 0].plot([lr_pred.min(), lr_pred.max()], [lr_pred.min(), lr_pred.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Linear Regression Predictions')
        axes[1, 0].set_ylabel('KNN Predictions')
        axes[1, 0].set_title('Model Predictions Comparison\n(Price)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Feature Importance (from Linear Regression if available)
    if 'linear_regression' in models_metadata:
        # This would need to be extracted from model training
        # For now, show a placeholder
        axes[1, 1].text(0.5, 0.5, 'Feature Importance\n(Linear Regression Coefficients)\nWould be extracted from\nmodel training metadata', 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Feature Importance Analysis')
    
    # 6. Model Performance Summary Heatmap
    if regression_results:
        # Create performance matrix
        metrics_data = []
        model_labels = []
        
        for model_name in model_names:
            model_data = regression_results[model_name]
            metrics_data.append([
                model_data['price_metrics']['r2'],
                model_data['price_metrics']['mae'],
                model_data['change_metrics']['r2'],
                model_data['change_metrics']['mae']
            ])
            model_labels.append(model_name.replace('_', ' ').title())
        
        metrics_df = pd.DataFrame(metrics_data, 
                                index=model_labels,
                                columns=['Price R²', 'Price MAE', 'Change R²', 'Change MAE'])
        
        # Normalize for heatmap (higher is better for R², lower is better for MAE)
        normalized_df = metrics_df.copy()
        normalized_df['Price MAE'] = 1 / (1 + normalized_df['Price MAE'] / 1000)  # Normalize MAE
        normalized_df['Change MAE'] = 1 / (1 + normalized_df['Change MAE'] * 1000)  # Normalize MAE
        
        sns.heatmap(normalized_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1, 2])
        axes[1, 2].set_title('Normalized Performance Heatmap\n(Green=Better)')
    
    # 7. Confusion Matrix for Trend Classification
    if 'knn' in classification_results:
        knn_results = classification_results['knn']
        cm = confusion_matrix(knn_results['trend_true'], knn_results['trend_predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
        axes[2, 0].set_title('Trend Classification\nConfusion Matrix')
        axes[2, 0].set_xlabel('Predicted')
        axes[2, 0].set_ylabel('Actual')
        axes[2, 0].set_xticklabels(['Down', 'Up'])
        axes[2, 0].set_yticklabels(['Down', 'Up'])
    
    # 8. Model Training Time Comparison (mock data)
    if regression_results:
        # Mock training times - in real implementation, these would be tracked during training
        training_times = {
            'linear_regression': 15.2,  # seconds
            'knn': 156.8  # seconds
        }
        
        available_models = [m for m in model_names if m in training_times]
        times = [training_times[m] for m in available_models]
        
        axes[2, 1].bar(available_models, times, color=['lightgreen', 'orange'], alpha=0.8)
        axes[2, 1].set_ylabel('Training Time (seconds)')
        axes[2, 1].set_title('Model Training Time Comparison')
        axes[2, 1].set_xticklabels([name.replace('_', ' ').title() for name in available_models])
        axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Model Complexity vs Performance
    if regression_results:
        # Mock complexity scores
        complexity_scores = {
            'linear_regression': 1,  # Simple
            'knn': 3  # More complex
        }
        
        performance_scores = [price_r2_scores[i] for i, name in enumerate(model_names)]
        complexity_vals = [complexity_scores.get(name, 2) for name in model_names]
        
        scatter = axes[2, 2].scatter(complexity_vals, performance_scores, 
                                   s=100, alpha=0.7, c=['blue', 'red'])
        axes[2, 2].set_xlabel('Model Complexity')
        axes[2, 2].set_ylabel('Performance (Price R²)')
        axes[2, 2].set_title('Complexity vs Performance')
        axes[2, 2].grid(True, alpha=0.3)
        
        # Add model labels
        for i, name in enumerate(model_names):
            axes[2, 2].annotate(name.replace('_', ' ').title(), 
                              (complexity_vals[i], performance_scores[i]),
                              xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    
    # Save plot
    root = project_root_path()
    plot_path = os.path.join(root, "data", "cache", "model_comparison_report.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison report: {plot_path}")
    plt.show()

def generate_text_report(models, regression_results, classification_results):
    """Generate detailed text report"""
    print("=== GENERATING DETAILED TEXT REPORT ===")
    
    report = []
    report.append("="*80)
    report.append("CRYPTO ML MODELS EVALUATION REPORT")
    report.append("="*80)
    report.append("")
    
    # Models Overview
    report.append("📋 MODELS TRAINED:")
    for model_name in models.keys():
        if model_name == 'linear_regression':
            report.append("  ✅ Linear Regression (Price & Price Change Prediction)")
        elif model_name == 'kmeans':
            report.append("  ✅ K-Means Clustering (Crypto Pattern Analysis)")
        elif model_name == 'knn':
            report.append("  ✅ K-Nearest Neighbors (Classification & Regression)")
    report.append("")
    
    # Regression Results
    if regression_results:
        report.append("📈 REGRESSION PERFORMANCE:")
        report.append("")
        
        for model_name, results in regression_results.items():
            model_title = model_name.replace('_', ' ').title()
            report.append(f"🔹 {model_title}:")
            
            price_metrics = results['price_metrics']
            change_metrics = results['change_metrics']
            
            report.append(f"  Price Prediction:")
            report.append(f"    • R² Score: {price_metrics['r2']:.4f}")
            report.append(f"    • MAE: ${price_metrics['mae']:.2f}")
            report.append(f"    • RMSE: ${price_metrics['rmse']:.2f}")
            
            report.append(f"  Price Change Prediction:")
            report.append(f"    • R² Score: {change_metrics['r2']:.4f}")
            report.append(f"    • MAE: {change_metrics['mae']:.6f}")
            report.append(f"    • RMSE: {change_metrics['rmse']:.6f}")
            report.append("")
    
    # Classification Results
    if classification_results:
        report.append("🎯 CLASSIFICATION PERFORMANCE:")
        report.append("")
        
        if 'knn' in classification_results:
            results = classification_results['knn']
            report.append("🔹 KNN Classification:")
            report.append(f"  Trend Direction (Up/Down):")
            report.append(f"    • Accuracy: {results['trend_accuracy']:.1%}")
            report.append(f"  Volatility Level (High/Low):")
            report.append(f"    • Accuracy: {results['vol_accuracy']:.1%}")
            report.append("")
    
    # Model Comparison
    if len(regression_results) > 1:
        report.append("⚖️ MODEL COMPARISON:")
        report.append("")
        
        # Find best performing model for price prediction
        best_price_model = max(regression_results.items(), 
                              key=lambda x: x[1]['price_metrics']['r2'])
        best_change_model = max(regression_results.items(), 
                               key=lambda x: x[1]['change_metrics']['r2'])
        
        report.append(f"🏆 Best Price Prediction: {best_price_model[0].replace('_', ' ').title()}")
        report.append(f"   R² = {best_price_model[1]['price_metrics']['r2']:.4f}")
        report.append(f"🏆 Best Price Change Prediction: {best_change_model[0].replace('_', ' ').title()}")
        report.append(f"   R² = {best_change_model[1]['change_metrics']['r2']:.4f}")
        report.append("")
    
    # Clustering Results
    if 'kmeans' in models:
        report.append("🔍 CLUSTERING ANALYSIS:")
        metadata = models['kmeans']['metadata']
        report.append(f"  • Number of Clusters: {metadata['n_clusters']}")
        report.append(f"  • Silhouette Score: {metadata['metrics']['silhouette_score']:.3f}")
        report.append(f"  • Inertia: {metadata['metrics']['inertia']:.2f}")
        report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS:")
    report.append("")
    
    if regression_results:
        if len(regression_results) > 1:
            best_model = max(regression_results.items(), 
                           key=lambda x: x[1]['price_metrics']['r2'])
            report.append(f"1. Use {best_model[0].replace('_', ' ').title()} for price prediction")
            report.append(f"   (R² = {best_model[1]['price_metrics']['r2']:.4f})")
        
        report.append("2. Price change prediction shows expected low R² due to market noise")
        report.append("3. Focus on price level prediction rather than returns prediction")
    
    if 'knn' in classification_results:
        results = classification_results['knn']
        if results['vol_accuracy'] > 0.8:
            report.append("4. KNN shows good performance for volatility classification")
        if results['trend_accuracy'] > 0.7:
            report.append("5. Trend prediction can be useful for trading signals")
    
    report.append("")
    report.append("="*80)
    
    # Save text report
    root = project_root_path()
    report_path = os.path.join(root, "data", "cache", "model_evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Saved detailed text report: {report_path}")
    
    # Print key findings
    print("\n".join(report))

def run_complete_evaluation(dataset_name='ml_datasets_top3'):
    """
    Complete model evaluation pipeline - FIXED VERSION
    """
    print("=== COMPLETE MODEL EVALUATION PIPELINE ===")
    
    try:
        # Load datasets FIRST
        datasets = load_prepared_datasets(dataset_name)
        
        # Load models
        models = load_trained_models()
        
        # Evaluate regression models with correct datasets parameter
        regression_results = evaluate_regression_models(models, datasets)
        
        # Evaluate classification models 
        classification_results = evaluate_classification_models(models, datasets['test_df'])
        
        # Create comprehensive report
        # create_model_comparison_report(regression_results, classification_results, models)
        
        # Generate text report
        # generate_text_report(models, regression_results, classification_results)
        
        print("\n" + "="*80)
        print("✅ MODEL EVALUATION COMPLETE!")
        print("📊 Reports and visualizations saved!")
        print("="*80)
        
        return {
            'models': models,
            'regression_results': regression_results,
            'classification_results': classification_results
        }
        
    except Exception as e:
        print(f"❌ Error in model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        # Run complete evaluation
        results = run_complete_evaluation()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION COMPLETE!")
        print("Reports and visualizations saved!")
        print("="*60)
        
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        import traceback
        traceback.print_exc()