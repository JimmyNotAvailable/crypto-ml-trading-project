#!/usr/bin/env python3
"""
🎯 MODEL PERFORMANCE ANALYZER & ALGORITHM SELECTOR
==================================================

Hệ thống tự động phân tích hiệu suất và chọn thuật toán tốt nhất:
- 📊 Phân tích metrics từ training jobs và model registry
- 🏆 Xếp hạng thuật toán theo hiệu suất
- 🎯 Tự động chọn thuật toán tốt nhất cho từng task
- 📈 Cung cấp khuyến nghị dựa trên dữ liệu thực tế
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

class ModelPerformanceAnalyzer:
    """
    🔍 Phân tích hiệu suất các thuật toán ML
    """
    
    def __init__(self, training_jobs_path: str = None, model_registry_path: str = None):
        # Tìm project root từ vị trí hiện tại
        current_path = Path(__file__).absolute()
        self.project_root = current_path.parent.parent.parent  # từ app/ml/ lên project root
        
        # Paths
        self.training_jobs_path = training_jobs_path or self.project_root / "training" / "training_jobs.json"
        self.model_registry_path = model_registry_path or self.project_root / "models" / "model_registry.json"
        
        # Load data
        self.training_jobs = self._load_training_jobs()
        self.model_registry = self._load_model_registry()
        
        # Performance metrics weights for scoring
        self.metric_weights = {
            'regression': {
                'r2': 0.5,           # R² score (higher = better)
                'mae': -0.3,         # Mean Absolute Error (lower = better)
                'rmse': -0.2         # Root Mean Square Error (lower = better)
            },
            'classification': {
                'accuracy': 0.6,     # Accuracy (higher = better)
                'precision': 0.2,    # Precision (higher = better)
                'recall': 0.2        # Recall (higher = better)
            },
            'clustering': {
                'silhouette_score': 0.7,    # Silhouette score (higher = better)
                'calinski_harabasz': 0.3    # Calinski-Harabasz index (higher = better)
            }
        }
    
    def _load_training_jobs(self) -> Dict:
        """Tải dữ liệu training jobs"""
        try:
            with open(self.training_jobs_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file: {self.training_jobs_path}")
            return {"jobs": {}}
        except Exception as e:
            print(f"❌ Lỗi khi tải training jobs: {e}")
            return {"jobs": {}}
    
    def _load_model_registry(self) -> Dict:
        """Tải dữ liệu model registry"""
        try:
            with open(self.model_registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file: {self.model_registry_path}")
            return {"models": {}}
        except Exception as e:
            print(f"❌ Lỗi khi tải model registry: {e}")
            return {"models": {}}
    
    def analyze_algorithm_performance(self) -> pd.DataFrame:
        """
        📊 Phân tích hiệu suất các thuật toán
        
        Returns:
            DataFrame chứa thống kê hiệu suất từng thuật toán
        """
        print("🔍 Phân tích hiệu suất các thuật toán...")
        
        algorithm_stats = {}
        
        # Phân tích từ training jobs
        for job_id, job_data in self.training_jobs.get('jobs', {}).items():
            if job_data.get('status') != 'completed':
                continue
                
            model_type = job_data.get('model_type', 'unknown')
            target_type = job_data.get('target_type', 'unknown')
            metrics = job_data.get('metrics', {})
            
            # Tạo key cho thuật toán + target
            algo_key = f"{model_type}_{target_type}"
            
            if algo_key not in algorithm_stats:
                algorithm_stats[algo_key] = {
                    'algorithm': model_type,
                    'target_type': target_type,
                    'jobs_count': 0,
                    'avg_r2': [],
                    'avg_mae': [],
                    'avg_rmse': [],
                    'avg_accuracy': [],
                    'latest_job': job_data.get('created_at'),
                    'best_r2': 0,
                    'best_mae': float('inf'),
                    'performance_scores': []
                }
            
            # Cập nhật thống kê
            stats = algorithm_stats[algo_key]
            stats['jobs_count'] += 1
            
            # Metrics từ test set (quan trọng nhất)
            test_metrics = metrics.get('test', {})
            if test_metrics:
                r2 = test_metrics.get('r2', 0)
                mae = test_metrics.get('mae', 0)
                rmse = test_metrics.get('rmse', 0)
                accuracy = test_metrics.get('accuracy', 0)
                
                stats['avg_r2'].append(r2)
                stats['avg_mae'].append(mae)
                stats['avg_rmse'].append(rmse)
                if accuracy > 0:
                    stats['avg_accuracy'].append(accuracy)
                
                # Cập nhật best scores
                if r2 > stats['best_r2']:
                    stats['best_r2'] = r2
                if mae > 0 and mae < stats['best_mae']:
                    stats['best_mae'] = mae
        
        # Tính toán trung bình và tạo DataFrame
        results = []
        for algo_key, stats in algorithm_stats.items():
            if stats['jobs_count'] == 0:
                continue
                
            result = {
                'algorithm': stats['algorithm'],
                'target_type': stats['target_type'],
                'jobs_count': stats['jobs_count'],
                'avg_r2': np.mean(stats['avg_r2']) if stats['avg_r2'] else 0,
                'avg_mae': np.mean(stats['avg_mae']) if stats['avg_mae'] else 0,
                'avg_rmse': np.mean(stats['avg_rmse']) if stats['avg_rmse'] else 0,
                'avg_accuracy': np.mean(stats['avg_accuracy']) if stats['avg_accuracy'] else 0,
                'best_r2': stats['best_r2'],
                'best_mae': stats['best_mae'] if stats['best_mae'] != float('inf') else 0,
                'latest_job': stats['latest_job']
            }
            
            # Tính performance score
            result['performance_score'] = self._calculate_performance_score(result)
            results.append(result)
        
        # Tạo DataFrame và sắp xếp theo performance score
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('performance_score', ascending=False).reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """
        🏆 Tính điểm hiệu suất tổng hợp
        
        Args:
            metrics: Dictionary chứa các metrics
            
        Returns:
            Performance score (0-100, càng cao càng tốt)
        """
        score = 0.0
        
        # Normalize R² score (0-100)
        r2_score = max(0, min(100, metrics.get('avg_r2', 0) * 100))
        score += r2_score * 0.5
        
        # Normalize MAE (convert to 0-100 scale, lower is better)
        mae = metrics.get('avg_mae', 0)
        if mae > 0:
            # For crypto prices, MAE around 20-50 is good, >100 is poor
            mae_score = max(0, 100 - (mae / 100) * 100)
            score += mae_score * 0.3
        
        # Accuracy for classification
        accuracy = metrics.get('avg_accuracy', 0)
        if accuracy > 0:
            score += accuracy * 100 * 0.6
        
        # Bonus for stability (multiple successful jobs)
        jobs_count = metrics.get('jobs_count', 1)
        stability_bonus = min(10, jobs_count * 2)  # Max 10 points
        score += stability_bonus
        
        return round(score, 2)
    
    def get_algorithm_ranking(self, target_type: str = None) -> pd.DataFrame:
        """
        🏅 Lấy xếp hạng thuật toán
        
        Args:
            target_type: Loại target cần lọc ('price', 'price_change', None cho tất cả)
            
        Returns:
            DataFrame đã được xếp hạng
        """
        df = self.analyze_algorithm_performance()
        
        if df.empty:
            print("⚠️ Không có dữ liệu để xếp hạng")
            return pd.DataFrame()
        
        if target_type:
            df = df[df['target_type'] == target_type].reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def recommend_best_algorithm(self, target_type: str, task_type: str = 'regression') -> Dict:
        """
        🎯 Khuyến nghị thuật toán tốt nhất
        
        Args:
            target_type: Loại target ('price', 'price_change', etc.)
            task_type: Loại task ('regression', 'classification', 'clustering')
            
        Returns:
            Dictionary chứa thông tin thuật toán được khuyến nghị
        """
        ranking = self.get_algorithm_ranking(target_type)
        
        if ranking.empty:
            return {
                'algorithm': 'linear_regression',  # Default fallback
                'confidence': 'low',
                'reason': 'Không có dữ liệu training để đánh giá',
                'performance_score': 0
            }
        
        best_algo = ranking.iloc[0]
        
        # Đánh giá confidence dựa trên performance score
        score = best_algo['performance_score']
        if score >= 80:
            confidence = 'very_high'
        elif score >= 60:
            confidence = 'high'
        elif score >= 40:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'algorithm': best_algo['algorithm'],
            'target_type': target_type,
            'performance_score': score,
            'confidence': confidence,
            'jobs_count': best_algo['jobs_count'],
            'avg_r2': best_algo['avg_r2'],
            'avg_mae': best_algo['avg_mae'],
            'rank': 1,
            'reason': f"Thuật toán tốt nhất với điểm số {score:.1f}/100 từ {best_algo['jobs_count']} lần training"
        }

class AlgorithmSelector:
    """
    🎯 Bộ chọn thuật toán tự động
    """
    
    def __init__(self):
        self.analyzer = ModelPerformanceAnalyzer()
    
    def select_best_algorithm_for_task(self, target_type: str, 
                                     min_confidence: str = 'medium') -> Dict:
        """
        🏆 Chọn thuật toán tốt nhất cho task cụ thể
        
        Args:
            target_type: Loại target cần dự đoán
            min_confidence: Confidence tối thiểu ('low', 'medium', 'high', 'very_high')
            
        Returns:
            Dictionary chứa thông tin thuật toán được chọn
        """
        recommendation = self.analyzer.recommend_best_algorithm(target_type)
        
        confidence_levels = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        min_level = confidence_levels.get(min_confidence, 2)
        current_level = confidence_levels.get(recommendation['confidence'], 1)
        
        if current_level >= min_level:
            return {
                'selected': True,
                'algorithm': recommendation['algorithm'],
                'confidence': recommendation['confidence'],
                'performance_score': recommendation['performance_score'],
                'reason': recommendation['reason']
            }
        else:
            return {
                'selected': False,
                'algorithm': 'linear_regression',  # Safe fallback
                'confidence': 'fallback',
                'performance_score': 0,
                'reason': f"Confidence {recommendation['confidence']} < yêu cầu {min_confidence}, sử dụng fallback"
            }
    
    def get_performance_report(self) -> str:
        """
        📊 Tạo báo cáo hiệu suất tổng quan
        
        Returns:
            String chứa báo cáo định dạng
        """
        ranking = self.analyzer.get_algorithm_ranking()
        
        if ranking.empty:
            return "⚠️ Không có dữ liệu để tạo báo cáo hiệu suất"
        
        report = "🏆 BÁO CÁO HIỆU SUẤT THUẬT TOÁN\n"
        report += "=" * 50 + "\n\n"
        
        # Top performers
        report += "🥇 TOP 5 THUẬT TOÁN TỐT NHẤT:\n"
        report += "-" * 30 + "\n"
        
        for i, row in ranking.head(5).iterrows():
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            report += f"{medal} {row['algorithm']} ({row['target_type']})\n"
            report += f"   📊 Điểm số: {row['performance_score']:.1f}/100\n"
            report += f"   📈 R²: {row['avg_r2']:.4f}\n"
            report += f"   📊 MAE: {row['avg_mae']:.2f}\n"
            report += f"   🔄 Số lần training: {row['jobs_count']}\n\n"
        
        # Performance by target type
        report += "📊 HIỆU SUẤT THEO LOẠI TARGET:\n"
        report += "-" * 30 + "\n"
        
        for target_type in ranking['target_type'].unique():
            target_data = ranking[ranking['target_type'] == target_type]
            best = target_data.iloc[0]
            report += f"🎯 {target_type.upper()}:\n"
            report += f"   ✅ Tốt nhất: {best['algorithm']} (điểm: {best['performance_score']:.1f})\n"
            report += f"   📊 Số thuật toán khả dụng: {len(target_data)}\n\n"
        
        return report

def main():
    """Demo function"""
    print("🎯 DEMO ALGORITHM SELECTOR")
    print("=" * 50)
    
    # Khởi tạo analyzer
    selector = AlgorithmSelector()
    
    # Test cho price prediction
    print("\n🔍 PHÂN TÍCH CHO DỰ ĐOÁN GIÁ:")
    price_selection = selector.select_best_algorithm_for_task('price')
    print(f"🏆 Thuật toán được chọn: {price_selection['algorithm']}")
    print(f"📊 Điểm số: {price_selection['performance_score']:.1f}")
    print(f"🎯 Confidence: {price_selection['confidence']}")
    print(f"💡 Lý do: {price_selection['reason']}")
    
    # Test cho price change prediction  
    print("\n🔍 PHÂN TÍCH CHO DỰ ĐOÁN THAY ĐỔI GIÁ:")
    change_selection = selector.select_best_algorithm_for_task('price_change')
    print(f"🏆 Thuật toán được chọn: {change_selection['algorithm']}")
    print(f"📊 Điểm số: {change_selection['performance_score']:.1f}")
    print(f"🎯 Confidence: {change_selection['confidence']}")
    print(f"💡 Lý do: {change_selection['reason']}")
    
    # Báo cáo tổng quan
    print("\n" + selector.get_performance_report())

if __name__ == "__main__":
    main()