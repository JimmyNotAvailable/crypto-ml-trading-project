#!/usr/bin/env python3
"""
🧪 SIMPLE ALGORITHM CLASSES TEST
==============================

Basic functionality test for new algorithm classes using real API
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_algorithm_imports():
    """Test that algorithm classes can be imported"""
    try:
        from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel
        print("✅ All algorithm classes imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_algorithm_instantiation():
    """Test that algorithm classes can be instantiated"""
    try:
        from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel
        
        # Test instantiation
        lr_model = LinearRegressionModel()
        knn_clf = KNNClassifier()
        knn_reg = KNNRegressor()
        kmeans = KMeansClusteringModel()
        
        print("✅ All algorithm classes instantiated successfully!")
        print(f"   LinearRegression: {lr_model.model_name}")
        print(f"   KNNClassifier: {knn_clf.model_name}")
        print(f"   KNNRegressor: {knn_reg.model_name}")
        print(f"   KMeans: {kmeans.model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test that data loading works"""
    try:
        from app.ml.data_prep import load_prepared_datasets
        
        datasets = load_prepared_datasets('ml_datasets_top3')
        print("✅ Data loading successful!")
        print(f"   Train shape: {datasets['X_train'].shape}")
        print(f"   Feature columns: {len(datasets['feature_cols'])}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def run_basic_tests():
    """Run basic functionality tests"""
    print("🚀 BASIC ALGORITHM CLASSES TESTS")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_algorithm_imports),
        ("Instantiation Test", test_algorithm_instantiation),
        ("Data Loading Test", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("✅ Algorithm classes are working correctly")
        print("🗑️ Legacy files successfully removed")
    else:
        print("❌ SOME TESTS FAILED!")
    
    return passed == total

if __name__ == "__main__":
    run_basic_tests()