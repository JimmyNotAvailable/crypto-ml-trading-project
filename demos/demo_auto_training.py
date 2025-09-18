#!/usr/bin/env python3
"""
🤖 AUTO TRAINING DEMO
===================
Submit thêm training jobs để test dashboard real-time
"""

import sys
import os
import time

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.services.trainer import enterprise_trainer

def main():
    print("🤖 Starting Auto Training Demo...")
    print("📊 Watch the dashboard for real-time updates!")
    print("🌐 Dashboard: http://localhost:8000/dashboard.html")
    print("-" * 50)
    
    # Submit multiple training jobs
    jobs = [
        ('linear_regression', 'auto_crypto_btc', 'price'),
        ('linear_regression', 'auto_crypto_eth', 'price_change'),
        ('knn', 'auto_knn_trend', 'price'),
        ('kmeans', 'auto_cluster_analysis', 'price')
    ]
    
    for i, (model_type, model_name, target_type) in enumerate(jobs):
        print(f"\n📝 Submitting job {i+1}/{len(jobs)}: {model_name}")
        
        job_id = enterprise_trainer.submit_training_job(
            model_type=model_type,
            model_name=model_name,
            target_type=target_type,
            created_by='Auto Demo'
        )
        
        print(f"✅ Job submitted: {job_id}")
        time.sleep(2)  # Delay để xem dashboard update
    
    print(f"\n🔄 Processing {len(jobs)} training jobs...")
    print("👀 Watch the dashboard Training Queue section!")
    
    # Process queue
    enterprise_trainer.process_queue()
    
    print("\n🎉 Demo completed! Check dashboard for results!")

if __name__ == "__main__":
    main()