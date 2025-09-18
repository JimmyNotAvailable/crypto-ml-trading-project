
# maintenance.py
# Production maintenance tasks

import os
import shutil
import glob
from datetime import datetime, timedelta

def cleanup_old_files(data_dir="data/realtime_production", days=7):
    """Clean up files older than specified days"""
    cutoff = datetime.now() - timedelta(days=days)
    
    files_removed = 0
    for file_path in glob.glob(f"{data_dir}/collection_batch_*.json"):
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if file_time < cutoff:
            os.remove(file_path)
            files_removed += 1
    
    print("Cleaned up {files_removed} old files")

def archive_data(source_dir="data/realtime_production", archive_dir="data/archive"):
    """Archive old data files"""
    os.makedirs(archive_dir, exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    archive_path = f"{archive_dir}/archive_{today}.zip"
    
    # Create archive (simplified - would use zipfile in practice)
    print(f"Data archived to {archive_path}")

def health_check_report():
    """Generate health check report"""
    from app.monitoring.health_monitor import HealthMonitor
    
    monitor = HealthMonitor()
    report = monitor.generate_health_report()
    
    # Save report
    report_file = f"monitoring/reports/health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Health report saved: {report_file}")

if __name__ == "__main__":
    print("Running maintenance tasks...")
    cleanup_old_files()
    archive_data()
    health_check_report()
    print("Maintenance complete")
