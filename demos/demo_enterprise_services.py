#!/usr/bin/env python3
"""
🏢 ENTERPRISE SERVICES DEMONSTRATION
===================================

Demonstrates the comprehensive enterprise architecture:
- ✅ Enterprise Data Store with versioning
- ✅ Automated Training Pipeline
- ✅ Model Registry Integration
- ✅ Job Queue Management
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.services.store import data_store
from app.services.trainer import enterprise_trainer
from app.ml.model_registry import model_registry
from app.ml.data_prep import load_prepared_datasets

def main():
    print("🏢 ENTERPRISE SERVICES DEMONSTRATION")
    print("===================================")
    
    try:
        # 1. Data Store Demo
        print("\n📊 ENTERPRISE DATA STORE")
        print("-" * 30)
        
        # Load some sample data
        datasets = load_prepared_datasets('ml_datasets_top3')
        sample_df = datasets['train_df'][:1000].copy()  # Sample for demo
        
        # Store dataset with versioning
        version_id = data_store.store_dataset(
            df=sample_df,
            dataset_name="crypto_sample",
            created_by="Demo User",
            description="Sample crypto data for enterprise demo"
        )
        
        print(f"✅ Stored dataset version: {version_id}")
        
        # List datasets
        datasets_list = data_store.list_datasets()
        print(f"📋 Total datasets in store: {len(datasets_list)}")
        
        # 2. Training Pipeline Demo
        print("\n🤖 AUTOMATED TRAINING PIPELINE")
        print("-" * 35)
        
        # Submit training jobs
        jobs = []
        
        # Submit Linear Regression jobs for different targets
        for target_type in ['price', 'price_change']:
            job_id = enterprise_trainer.submit_training_job(
                model_type='linear_regression',
                model_name=f'enterprise_crypto_{target_type}',
                dataset_name='ml_datasets_top3',
                target_type=target_type,
                created_by='Enterprise Demo'
            )
            jobs.append(job_id)
            print(f"📝 Submitted job: {job_id}")
        
        # List pending jobs
        pending_jobs = enterprise_trainer.list_jobs(status='pending')
        print(f"\n📋 Pending jobs: {len(pending_jobs)}")
        
        # Process the queue
        print("\n🔄 Processing training queue...")
        enterprise_trainer.process_queue()
        
        # Show completed jobs
        completed_jobs = enterprise_trainer.list_jobs(status='completed')
        print(f"\n✅ Completed jobs: {len(completed_jobs)}")
        
        if not completed_jobs.empty:
            for _, job in completed_jobs.iterrows():
                print(f"   🔸 {job['model_name']}: {job['model_type']} -> {job['model_id']}")
                print(f"      Duration: {job['duration']}")
        
        # 3. Model Registry Integration
        print("\n📦 MODEL REGISTRY STATUS")
        print("-" * 25)
        
        models_df = model_registry.list_models()
        print(f"📊 Total models in registry: {len(models_df)}")
        
        if not models_df.empty:
            # Show latest models
            latest_models = models_df.head(3)
            for _, model in latest_models.iterrows():
                print(f"🔸 {model['model_name']} v{model['version']}")
                print(f"   Performance: R² = {model['test_r2']:.4f}")
                print(f"   Created: {model['timestamp'][:19]}")
        
        # 4. Enterprise Statistics
        print("\n📈 ENTERPRISE STATISTICS")
        print("-" * 25)
        
        total_jobs = enterprise_trainer.list_jobs()
        print(f"🎯 Total training jobs: {len(total_jobs)}")
        print(f"✅ Successful jobs: {len(total_jobs[total_jobs['status'] == 'completed'])}")
        print(f"❌ Failed jobs: {len(total_jobs[total_jobs['status'] == 'failed'])}")
        print(f"⏳ Pending jobs: {len(total_jobs[total_jobs['status'] == 'pending'])}")
        
        print(f"\n💾 Data store location: {data_store.store_path}")
        print(f"🏭 Training workspace: {enterprise_trainer.trainer_path}")
        print(f"📦 Model registry: {model_registry.registry_path}")
        
        print("\n🎉 Enterprise architecture demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()