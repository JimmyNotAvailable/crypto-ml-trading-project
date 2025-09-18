@echo off
echo Starting Crypto Data Collector Production
cd /d "%~dp0"

echo Checking system health...
python app/monitoring/health_monitor.py

echo Starting continuous collection...
python scripts/continuous_collector.py --hours 12 --mongodb

pause
