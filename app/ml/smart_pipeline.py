#!/usr/bin/env python3
"""
🤖 SMART TRAINING PIPELINE
=========================

Pipeline training thông minh tự động chọn thuật toán tốt nhất:
- 🎯 Phân tích hiệu suất từ lịch sử training
- 🏆 Tự động chọn thuật toán tốt nhất cho từng task
- 📊 Báo cáo chi tiết về lựa chọn
- 🔄 Fallback an toàn khi không có đủ dữ liệu
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.ml.algorithm_selector import AlgorithmSelector
from app.ml.algorithms import LinearRegressionModel, KNNClassifier, KNNRegressor, KMeansClusteringModel, RandomForestModel
from app.ml.data_prep import load_prepared_datasets
import pandas as pd

class SmartTrainingPipeline:
    """
    🧠 Pipeline training thông minh với auto algorithm selection
    """
    
    def __init__(self):
        self.selector = AlgorithmSelector()
        self.supported_algorithms = {
            'linear_regression': LinearRegressionModel,
            'knn_classifier': KNNClassifier,
            'knn_regressor': KNNRegressor,
            'knn': KNNRegressor,  # Alias
            'kmeans': KMeansClusteringModel,
            'random_forest_regression': RandomForestModel,
            'random_forest_classification': RandomForestModel,
            'random_forest': RandomForestModel  # Default to regression
        }
    
    def _convert_datasets_format(self, datasets_raw: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        🔄 Convert dataset format từ cache format sang algorithm format
        
        Args:
            datasets_raw: Raw datasets từ cache (X_train, y_train, etc.)
            
        Returns:
            Dict với format mà algorithms mong đợi ('train', 'test' DataFrames)
        """
        return {
            'train': datasets_raw['train_df'],
            'test': datasets_raw['test_df']
        }
    
    def train_with_best_algorithm(self, target_type: str, 
                                  dataset_name: str = 'ml_datasets_top3',
                                  min_confidence: str = 'medium',
                                  force_algorithm: str = None) -> Dict[str, Any]:
        """
        🎯 Training với thuật toán tốt nhất được tự động chọn
        
        Args:
            target_type: Loại target ('price', 'price_change', etc.)
            dataset_name: Tên dataset để sử dụng
            min_confidence: Confidence tối thiểu để sử dụng recommendation
            force_algorithm: Bắt buộc sử dụng thuật toán cụ thể (bỏ qua auto selection)
            
        Returns:
            Dictionary chứa kết quả training và thông tin algorithm selection
        """
        print(f"🤖 SMART TRAINING PIPELINE - TARGET: {target_type}")
        print("=" * 60)
        
        # Load data
        print("📊 Đang tải dữ liệu...")
        try:
            datasets_raw = load_prepared_datasets(dataset_name)
            print(f"✅ Tải dữ liệu thành công: {len(datasets_raw['X_train'])} mẫu training")
            
            # Convert format for algorithms (they expect 'train', 'test' keys with DataFrames)
            datasets = self._convert_datasets_format(datasets_raw)
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Lỗi khi tải dữ liệu: {str(e)}",
                'algorithm_selection': None,
                'training_result': None
            }
        
        # Algorithm Selection
        if force_algorithm:
            print(f"🔒 Bắt buộc sử dụng thuật toán: {force_algorithm}")
            algorithm_choice = {
                'selected': True,
                'algorithm': force_algorithm,
                'confidence': 'forced',
                'performance_score': -1,
                'reason': f"Bắt buộc sử dụng {force_algorithm}"
            }
        else:
            print("🎯 Đang phân tích và chọn thuật toán tốt nhất...")
            algorithm_choice = self.selector.select_best_algorithm_for_task(
                target_type, min_confidence
            )
        
        print(f"🏆 Thuật toán được chọn: {algorithm_choice['algorithm']}")
        print(f"📊 Điểm hiệu suất: {algorithm_choice['performance_score']:.1f}/100")
        print(f"🎯 Confidence: {algorithm_choice['confidence']}")
        print(f"💡 Lý do: {algorithm_choice['reason']}")
        
        # Training
        selected_algorithm = algorithm_choice['algorithm']
        
        if selected_algorithm not in self.supported_algorithms:
            return {
                'success': False,
                'error': f"Thuật toán {selected_algorithm} không được hỗ trợ",
                'algorithm_selection': algorithm_choice,
                'training_result': None
            }
        
        print(f"\n🚀 Bắt đầu training với {selected_algorithm}...")
        
        try:
            # Khởi tạo model dựa trên thuật toán được chọn
            model_class = self.supported_algorithms[selected_algorithm]
            
            if selected_algorithm == 'linear_regression':
                model = model_class(target_type=target_type)
            elif selected_algorithm in ['knn_classifier', 'knn_regressor', 'knn']:
                model = model_class()
            elif selected_algorithm == 'kmeans':
                model = model_class()
            elif selected_algorithm == 'random_forest_regression':
                model = model_class(task_type='regression', target_type=target_type)
            elif selected_algorithm == 'random_forest_classification':
                model = model_class(task_type='classification', target_type=target_type)
            elif selected_algorithm == 'random_forest':
                # Default to regression for price-related targets
                task_type = 'regression' if target_type in ['price', 'volume', 'market_cap'] else 'classification'
                model = model_class(task_type=task_type, target_type=target_type)
            else:
                model = model_class()
            
            # Training
            start_time = datetime.now()
            training_metrics = model.train(datasets)
            end_time = datetime.now()
            
            training_duration = (end_time - start_time).total_seconds()
            
            print(f"✅ Training hoàn thành trong {training_duration:.2f}s")
            
            # Save model
            model_filename = f"smart_pipeline_{target_type}_{selected_algorithm}"
            model_path = model.save_model(model_filename)
            
            print(f"💾 Model đã lưu: {model_path}")
            
            return {
                'success': True,
                'algorithm_selection': algorithm_choice,
                'training_result': {
                    'algorithm': selected_algorithm,
                    'target_type': target_type,
                    'metrics': training_metrics,
                    'training_duration': training_duration,
                    'model_path': model_path,
                    'dataset_name': dataset_name
                },
                'model': model
            }
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình training: {str(e)}")
            return {
                'success': False,
                'error': f"Lỗi training: {str(e)}",
                'algorithm_selection': algorithm_choice,
                'training_result': None
            }
    
    def train_multiple_targets(self, target_types: list,
                              dataset_name: str = 'ml_datasets_top3',
                              min_confidence: str = 'medium') -> Dict[str, Any]:
        """
        🎯 Training cho nhiều targets với auto algorithm selection
        
        Args:
            target_types: List các loại target cần training
            dataset_name: Tên dataset
            min_confidence: Confidence tối thiểu
            
        Returns:
            Dictionary chứa kết quả cho tất cả targets
        """
        print("🎪 SMART MULTI-TARGET TRAINING")
        print("=" * 60)
        
        results = {}
        successful_trainings = 0
        
        for target_type in target_types:
            print(f"\n📍 TRAINING TARGET: {target_type}")
            print("-" * 40)
            
            result = self.train_with_best_algorithm(
                target_type=target_type,
                dataset_name=dataset_name,
                min_confidence=min_confidence
            )
            
            results[target_type] = result
            
            if result['success']:
                successful_trainings += 1
                print(f"✅ {target_type}: Thành công")
            else:
                print(f"❌ {target_type}: Thất bại - {result.get('error', 'Unknown error')}")
        
        # Summary
        print(f"\n📊 TÓM TẮT KẾT QUẢ:")
        print(f"✅ Thành công: {successful_trainings}/{len(target_types)}")
        print(f"❌ Thất bại: {len(target_types) - successful_trainings}/{len(target_types)}")
        
        return {
            'total_targets': len(target_types),
            'successful_trainings': successful_trainings,
            'failed_trainings': len(target_types) - successful_trainings,
            'results': results
        }
    
    def get_algorithm_recommendations(self, target_types: list = None) -> Dict[str, Any]:
        """
        🎯 Lấy khuyến nghị thuật toán cho các targets
        
        Args:
            target_types: List targets cần phân tích (None = tất cả)
            
        Returns:
            Dictionary chứa khuyến nghị cho từng target
        """
        if target_types is None:
            target_types = ['price', 'price_change']
        
        recommendations = {}
        
        print("🎯 KHUYẾN NGHỊ THUẬT TOÁN")
        print("=" * 40)
        
        for target_type in target_types:
            rec = self.selector.select_best_algorithm_for_task(target_type)
            recommendations[target_type] = rec
            
            print(f"\n📊 {target_type.upper()}:")
            print(f"   🏆 Thuật toán: {rec['algorithm']}")
            print(f"   📈 Điểm số: {rec['performance_score']:.1f}/100")
            print(f"   🎯 Confidence: {rec['confidence']}")
        
        return recommendations

def demo_smart_pipeline():
    """Demo Smart Training Pipeline"""
    pipeline = SmartTrainingPipeline()
    
    print("🤖 DEMO SMART TRAINING PIPELINE")
    print("=" * 50)
    
    # 1. Xem khuyến nghị thuật toán
    print("\n1️⃣ PHÂN TÍCH KHUYẾN NGHỊ:")
    recommendations = pipeline.get_algorithm_recommendations(['price', 'price_change'])
    
    # 2. Training single target với auto selection
    print("\n2️⃣ TRAINING SINGLE TARGET (AUTO SELECTION):")
    price_result = pipeline.train_with_best_algorithm('price')
    
    if price_result['success']:
        training_info = price_result['training_result']
        print(f"\n✅ KẾT QUẢ TRAINING:")
        print(f"   🎯 Target: {training_info['target_type']}")
        print(f"   🏆 Algorithm: {training_info['algorithm']}")
        print(f"   ⏱️ Thời gian: {training_info['training_duration']:.2f}s")
        print(f"   💾 Model path: {training_info['model_path']}")
    
    # 3. Training với force algorithm
    print("\n3️⃣ TRAINING VỚI FORCE ALGORITHM:")
    forced_result = pipeline.train_with_best_algorithm(
        target_type='price_change',
        force_algorithm='linear_regression'
    )
    
    print("\n🎉 DEMO HOÀN TẤT!")

if __name__ == "__main__":
    demo_smart_pipeline()