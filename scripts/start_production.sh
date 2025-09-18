#!/bin/bash
echo "Starting Crypto Data Collector Production"
cd "$(dirname "$0")"

echo "Checking system health..."
python3 app/monitoring/health_monitor.py

echo "Starting continuous collection..."
python3 scripts/continuous_collector.py --hours 12 --mongodb
