#!/usr/bin/env python3
"""
🎪 DEMO SMART MULTI-TARGET TRAINING
================================

Demo pipeline training thông minh cho nhiều targets cùng lúc
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.absolute())
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.ml.smart_pipeline import SmartTrainingPipeline

def demo_multi_target():
    """Demo multi-target training với Smart Pipeline"""
    
    print("🎪 SMART MULTI-TARGET TRAINING DEMO")
    print("=" * 60)
    
    pipeline = SmartTrainingPipeline()
    
    # 1. Khuyến nghị cho tất cả targets
    print("\n🎯 1. PHÂN TÍCH KHUYẾN NGHỊ CHO TẤT CẢ TARGETS:")
    targets = ['price', 'price_change']
    recommendations = pipeline.get_algorithm_recommendations(targets)
    
    # 2. Multi-target training
    print("\n🚀 2. TRAINING NHIỀU TARGETS CÙNG LÚC:")
    results = pipeline.train_multiple_targets(
        target_types=targets,
        min_confidence='low'  # Accept low confidence cho demo
    )
    
    # 3. Báo cáo chi tiết
    print(f"\n📊 3. BÁO CÁO CHI TIẾT:")
    print(f"┌─────────────────────────────────────────┐")
    print(f"│ 📈 MULTI-TARGET TRAINING SUMMARY       │")
    print(f"├─────────────────────────────────────────┤")
    print(f"│ Total targets: {results['total_targets']:<2}                      │")
    print(f"│ ✅ Successful: {results['successful_trainings']:<2}                      │")
    print(f"│ ❌ Failed: {results['failed_trainings']:<2}                          │")
    print(f"│ Success rate: {(results['successful_trainings']/results['total_targets']*100):.1f}%                  │")
    print(f"└─────────────────────────────────────────┘")
    
    # 4. Chi tiết từng target
    print(f"\n📋 4. CHI TIẾT TỪNG TARGET:")
    for target, result in results['results'].items():
        print(f"\n📊 {target.upper()}:")
        if result['success']:
            training_info = result['training_result']
            algo_info = result['algorithm_selection']
            print(f"   ✅ Status: Thành công")
            print(f"   🏆 Algorithm: {training_info['algorithm']}")
            print(f"   📈 Performance Score: {algo_info['performance_score']:.1f}/100")
            print(f"   🎯 Confidence: {algo_info['confidence']}")
            print(f"   ⏱️ Training Time: {training_info['training_duration']:.2f}s")
            print(f"   💾 Model Path: {os.path.basename(training_info['model_path'])}")
        else:
            print(f"   ❌ Status: Thất bại")
            print(f"   🔍 Error: {result['error']}")
    
    print(f"\n🎉 MULTI-TARGET DEMO HOÀN TẤT!")
    print(f"🎯 Pipeline có thể train tự động cho bất kỳ targets nào!")

if __name__ == "__main__":
    demo_multi_target()