# train_all.py
# Main orchestration script cho toàn bộ ML pipeline - 34 coins ready

import os
import sys
import time
from datetime import datetime

# Import all ML modules
sys.path.append(os.path.dirname(__file__))
from data_prep import load_prepared_datasets, prepare_ml_datasets, save_prepared_datasets, load_features_data

# Import new algorithm classes
from algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel

from evaluate import run_complete_evaluation

def project_root_path():
    """Get project root path"""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

def setup_directories():
    """Ensure all required directories exist"""
    root = project_root_path()
    dirs_to_create = [
        os.path.join(root, "models"),
        os.path.join(root, "data", "cache"),
        os.path.join(root, "reports")
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")

def train_all_models(config):
    """
    Train all ML models with given configuration
    
    Args:
        config (dict): Training configuration
    """
    dataset_name = config['dataset_name']
    
    print(f"\n🚀 TRAINING ALL MODELS")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🎯 Target: {config['n_symbols']} symbols")
    
    results = {}
    start_time = time.time()
    
    try:
        # Load datasets
        datasets = load_prepared_datasets(dataset_name)
        
        # 1. LINEAR REGRESSION
        print(f"\n{'='*60}")
        print("1️⃣  TRAINING LINEAR REGRESSION")
        print(f"{'='*60}")
        
        lr_start = time.time()
        
        # Train price prediction model
        lr_price = LinearRegressionModel(target_type='price', normalize_features=True)
        price_metrics = lr_price.train(datasets)
        price_model_path = lr_price.save_model("linreg_price")
        
        # Train price change prediction model
        lr_change = LinearRegressionModel(target_type='price_change', normalize_features=True)
        change_metrics = lr_change.train(datasets)
        change_model_path = lr_change.save_model("linreg_price_change")
        
        lr_time = time.time() - lr_start
        
        lr_results = {
            'price_prediction': {
                'metrics': price_metrics,
                'model_path': price_model_path
            },
            'price_change_prediction': {
                'metrics': change_metrics,
                'model_path': change_model_path
            }
        }
        
        results['linear_regression'] = {
            'status': 'success',
            'training_time': lr_time,
            'results': lr_results
        }
        print(f"✅ Linear Regression completed in {lr_time:.1f}s")
        
        # 2. K-MEANS CLUSTERING
        print(f"\n{'='*60}")
        print("2️⃣  TRAINING K-MEANS CLUSTERING")
        print(f"{'='*60}")
        
        kmeans_start = time.time()
        
        # Train clustering model
        kmeans_model = KMeansClusteringModel(auto_tune=True, max_clusters=8)
        clustering_metrics = kmeans_model.train(datasets)
        clustering_model_path = kmeans_model.save_model("kmeans_crypto")
        
        kmeans_time = time.time() - kmeans_start
        
        kmeans_results = {
            'metrics': clustering_metrics,
            'model_path': clustering_model_path,
            'n_clusters': kmeans_model.n_clusters,
            'cluster_centers': kmeans_model.get_cluster_centers().to_dict()
        }
        
        results['kmeans'] = {
            'status': 'success',
            'training_time': kmeans_time,
            'results': kmeans_results
        }
        print(f"✅ K-Means completed in {kmeans_time:.1f}s")
        
        # 3. KNN MODELS
        print(f"\n{'='*60}")
        print("3️⃣  TRAINING KNN MODELS")
        print(f"{'='*60}")
        
        knn_start = time.time()
        
        # Train KNN Classifier for trend prediction
        knn_classifier = KNNClassifier(auto_tune=True)
        classifier_metrics = knn_classifier.train(datasets)
        classifier_model_path = knn_classifier.save_model("knn_classifier")
        
        # Train KNN Regressor for price prediction
        knn_regressor = KNNRegressor(target_type='price', auto_tune=True)
        regressor_metrics = knn_regressor.train(datasets)
        regressor_model_path = knn_regressor.save_model("knn_regressor")
        
        knn_time = time.time() - knn_start
        
        knn_results = {
            'classifier': {
                'metrics': classifier_metrics,
                'model_path': classifier_model_path
            },
            'regressor': {
                'metrics': regressor_metrics,
                'model_path': regressor_model_path
            }
        }
        
        results['knn'] = {
            'status': 'success',
            'training_time': knn_time,
            'results': knn_results
        }
        print(f"✅ KNN completed in {knn_time:.1f}s")
        
        total_time = time.time() - start_time
        
        print(f"\n🎯 TRAINING SUMMARY:")
        print(f"⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
        
        successful_models = [name for name, result in results.items() if result['status'] == 'success']
        failed_models = [name for name, result in results.items() if result['status'] == 'failed']
        
        print(f"✅ Successful models ({len(successful_models)}): {', '.join(successful_models)}")
        if failed_models:
            print(f"❌ Failed models ({len(failed_models)}): {', '.join(failed_models)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Critical error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return results

def run_evaluation(dataset_name):
    """Run comprehensive model evaluation"""
    print(f"\n{'='*60}")
    print("4️⃣  RUNNING MODEL EVALUATION")
    print(f"{'='*60}")
    
    try:
        eval_results = run_complete_evaluation(dataset_name)
        if eval_results:
            print("✅ Model evaluation completed successfully!")
            return eval_results
        else:
            print("❌ Model evaluation failed!")
            return None
            
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_pipeline_report(config, training_results, eval_results):
    """Generate comprehensive pipeline report"""
    root = project_root_path()
    report_path = os.path.join(root, "reports", f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CRYPTO ML PIPELINE EXECUTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📊 Dataset: {config['dataset_name']}\n")
        f.write(f"🪙 Number of symbols: {config['n_symbols']}\n")
        f.write(f"🎯 Configuration: {config['description']}\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        for model_name, result in training_results.items():
            status = "✅ SUCCESS" if result['status'] == 'success' else "❌ FAILED"
            f.write(f"{model_name.upper()}: {status} ({result['training_time']:.1f}s)\n")
        
        if eval_results:
            f.write("\nEVALUATION SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            if 'regression_results' in eval_results:
                f.write("Regression Performance:\n")
                for model, metrics in eval_results['regression_results'].items():
                    if 'price' in metrics:
                        f.write(f"  {model}: R² = {metrics['price']['r2']:.4f}\n")
            
            if 'classification_results' in eval_results:
                f.write("Classification Performance:\n")
                for model, metrics in eval_results['classification_results'].items():
                    if 'accuracy' in metrics:
                        f.write(f"  {model}: Accuracy = {metrics['accuracy']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"📄 Pipeline report saved: {report_path}")
    return report_path

def main():
    """
    Main execution function for complete ML pipeline
    """
    print("🚀 CRYPTO ML PIPELINE - COMPLETE EXECUTION")
    print("="*80)
    
    # Setup
    setup_directories()
    
    # Configuration options
    configs = {
        'test': {
            'dataset_name': 'ml_datasets_top3',
            'n_symbols': 3,
            'symbols': None,
            'description': 'Test run with top 3 symbols'
        },
        'small': {
            'dataset_name': 'ml_datasets_top5',
            'n_symbols': 5,
            'symbols': None,
            'description': 'Small run with top 5 symbols'
        },
        'full': {
            'dataset_name': 'ml_datasets_top34',
            'n_symbols': 34,
            'symbols': None,
            'description': 'Full run with top 34 symbols'
        },
        'custom': {
            'dataset_name': 'ml_datasets_custom',
            'n_symbols': 'custom',
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT'],
            'description': 'Custom symbols selection'
        }
    }
    
    # Select configuration
    config_name = 'test'  # Change this: 'test', 'small', 'full', 'custom'
    config = configs[config_name]
    
    print(f"📋 CONFIGURATION: {config['description']}")
    print(f"🎯 Target: {config['n_symbols']} symbols")
    
    # Check if datasets exist, if not prepare them
    try:
        datasets = load_prepared_datasets(config['dataset_name'])
        print(f"✅ Found existing datasets: {config['dataset_name']}")
    except FileNotFoundError:
        print(f"📊 Preparing new datasets: {config['dataset_name']}")
        
        if config_name == 'full':
            df = load_features_data(top_n=34)
        elif config_name == 'custom':
            df = load_features_data(symbols=config['symbols'])
        else:
            df = load_features_data(top_n=config['n_symbols'])
        
        datasets = prepare_ml_datasets(df)
        save_prepared_datasets(datasets, config['dataset_name'])
        print(f"✅ Datasets prepared and saved!")
    
    # Execute pipeline
    pipeline_start = time.time()
    
    # 1. Train all models
    training_results = train_all_models(config)
    
    # 2. Run evaluation
    eval_results = run_evaluation(config['dataset_name'])
    
    # 3. Generate report
    report_path = generate_pipeline_report(config, training_results, eval_results)
    
    total_pipeline_time = time.time() - pipeline_start
    
    # Final summary
    print(f"\n" + "="*80)
    print("🎉 PIPELINE EXECUTION COMPLETE!")
    print(f"⏱️  Total execution time: {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f}m)")
    print(f"📄 Report saved: {report_path}")
    
    successful_models = [name for name, result in training_results.items() if result['status'] == 'success']
    print(f"✅ Successfully trained: {len(successful_models)}/{len(training_results)} models")
    
    if eval_results:
        print("✅ Evaluation completed successfully")
    else:
        print("⚠️  Evaluation completed with issues")
    
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical pipeline error: {e}")
        import traceback
        traceback.print_exc()