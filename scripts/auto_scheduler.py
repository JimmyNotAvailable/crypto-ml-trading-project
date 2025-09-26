#!/usr/bin/env python3
"""
🤖 CRYPTO ML AUTO SCHEDULER
===========================
Scheduler tự động để train models theo lịch
"""

import os
import sys
import time
from datetime import datetime

# Add project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from app.services.trainer import enterprise_trainer

def daily_training():
    """Train models hàng ngày"""
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting daily training...")
    
    # Submit training jobs
    jobs = [
        ('linear_regression', f'daily_crypto_price_{datetime.now().strftime("%Y%m%d")}', 'price'),
        ('linear_regression', f'daily_crypto_change_{datetime.now().strftime("%Y%m%d")}', 'price_change'),
        ('knn', f'daily_knn_{datetime.now().strftime("%Y%m%d")}', 'price'),
    ]
    
    for model_type, model_name, target_type in jobs:
        job_id = enterprise_trainer.submit_training_job(
            model_type=model_type,
            model_name=model_name,
            target_type=target_type,
            created_by='Auto Scheduler'
        )
        print(f"✅ Submitted: {job_id}")
    
    # Process queue
    enterprise_trainer.process_queue()
    print("🎯 Daily training completed!")

def weekly_full_training():
    """Train toàn bộ models hàng tuần"""
    print(f"🗓️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Starting weekly full training...")
    
    jobs = [
        ('linear_regression', f'weekly_crypto_price_{datetime.now().strftime("%Y%W")}', 'price'),
        ('linear_regression', f'weekly_crypto_change_{datetime.now().strftime("%Y%W")}', 'price_change'),
        ('knn', f'weekly_knn_{datetime.now().strftime("%Y%W")}', 'price'),
        ('kmeans', f'weekly_clustering_{datetime.now().strftime("%Y%W")}', 'price'),
    ]
    
    for model_type, model_name, target_type in jobs:
        job_id = enterprise_trainer.submit_training_job(
            model_type=model_type,
            model_name=model_name,
            target_type=target_type,
            created_by='Weekly Scheduler'
        )
        print(f"✅ Submitted: {job_id}")
    
    enterprise_trainer.process_queue()
    print("🎯 Weekly training completed!")

def main():
    print("🤖 CRYPTO ML AUTO SCHEDULER STARTED")
    print("=" * 40)
    print("📅 Schedule:")
    print("  - Daily training: 02:00 AM")
    print("  - Weekly training: Sunday 03:00 AM")
    print("🌐 Dashboard: http://localhost:8000/dashboard.html")
    print("⚡ Press Ctrl+C to stop scheduler")
    print("-" * 40)
    
    # Import schedule lazily to avoid hard dependency at import time
    try:
        import schedule  # type: ignore
    except Exception as e:
        print(f"⚠️ 'schedule' package is not installed. Install with: pip install schedule. Running once now. Details: {e}")
        # Fallback: run both tasks once and exit
        try:
            daily_training()
            weekly_full_training()
        except Exception as inner:
            print(f"❌ Training execution error: {inner}")
        return

    # Schedule daily training at 2 AM
    schedule.every().day.at("02:00").do(daily_training)

    # Schedule weekly training on Sunday at 3 AM
    schedule.every().sunday.at("03:00").do(weekly_full_training)
    
    # For testing - run every 5 minutes
    # schedule.every(5).minutes.do(daily_training)
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n🛑 Scheduler stopped by user")
            break
        except Exception as e:
            print(f"❌ Scheduler error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()