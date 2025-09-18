
# Crypto Data Collector - Production Setup

## 🏗️ System Architecture

```
crypto-project/
├── app/
│   ├── data_collector/          # Core data collection
│   ├── database/                # MongoDB integration  
│   ├── monitoring/              # Health monitoring
│   └── ml/                      # ML models
├── config/                      # Configuration files
├── scripts/                     # Operational scripts
├── data/                        # Data storage
└── logs/                        # Log files
```

## 🚀 Quick Start

### Windows
```cmd
start_production.bat
```

### Linux/Mac
```bash
./start_production.sh
```

## 📊 Manual Operations

### Start Data Collection
```bash
python scripts/continuous_collector.py --hours 12 --mongodb
```

### Health Check
```bash
python app/monitoring/health_monitor.py
```

### Configuration
```bash
python config/production_config.py
```

## 🔧 Configuration

### Environment Variables
```bash
export MONGODB_URI="mongodb://localhost:27017/"
export MONGODB_DATABASE="crypto"
export LOG_LEVEL="INFO"
```

### Collection Settings
- **Symbols**: 34 cryptocurrencies
- **Interval**: 60 seconds
- **Duration**: Configurable (1-12+ hours)
- **Storage**: MongoDB + File backup

## 📈 Monitoring

### Health Checks
- System resources (CPU, Memory, Disk)
- Data freshness and quality
- API connectivity and performance
- MongoDB connection status

### Alerts
- Data older than 10 minutes
- Success rate below 95%
- High system resource usage
- API response time issues

## 🗄️ Data Management

### Storage
- **Real-time**: `data/realtime_production/`
- **Archive**: `data/archive/`
- **Logs**: `logs/`

### Backup Strategy
- File-based backup (always enabled)
- MongoDB primary storage
- Daily archive rotation
- 7-day retention policy

## 🔧 Maintenance

### Daily Tasks
```bash
python scripts/maintenance.py
```

### Manual Cleanup
```bash
# Clean old files (7 days+)
find data/realtime_production -name "collection_batch_*.json" -mtime +7 -delete

# Archive data
python scripts/maintenance.py
```

## 📊 Performance Metrics

### Current Performance
- **Success Rate**: 100%
- **Collection Time**: ~16 seconds/cycle
- **Data Size**: ~35KB/batch
- **API Calls**: ~68/minute
- **Memory Usage**: ~35MB

### Scaling Considerations
- Rate limits: 1000 requests/minute (Binance)
- Storage: ~200MB/day (continuous)
- CPU: Low impact (<20% during collection)

## 🚨 Troubleshooting

### Common Issues
1. **MongoDB Connection Failed**
   - Check MongoDB service status
   - Verify connection string
   - Falls back to file storage

2. **API Rate Limits**
   - Automatic rate limiting implemented
   - Increase intervals if needed

3. **High Memory Usage**
   - Normal during collection cycles
   - Monitor for memory leaks

4. **Stale Data**
   - Check collector process status
   - Verify API connectivity

## 🎯 Next Steps

1. **Discord Bot Integration** 
2. **Web Dashboard**
3. **Alert System**
4. **Auto-scaling**

## 📞 Support

Check health monitoring for system status and alerts.
