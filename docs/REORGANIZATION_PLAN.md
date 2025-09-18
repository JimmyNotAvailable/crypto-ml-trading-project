# PROJECT REORGANIZATION PLAN

## 📁 CURRENT ISSUES
Files misplaced in root directory that should be organized properly:

### 🚨 Root Level Files to Relocate:
1. **Data Quality & Analysis Scripts:**
   - debug_data.py → scripts/analysis/
   - data_quality_fix.py → scripts/data_processing/
   - data_quality_fix_v2.py → scripts/data_processing/
   - retrain_models.py → scripts/ml/
   - validate_models.py → scripts/ml/

2. **Testing & Demo Files:**
   - test_random_forest.py → tests/ml/
   - demo_smart_multi_target.py → demos/ml/

3. **Documentation & Reports:**
   - BUG_FIX_REPORT.md → docs/reports/
   - DATA_QUALITY_FINAL_REPORT.md → docs/reports/
   - DEBUG_DATA_ANALYSIS.md → docs/reports/
   - IMPORT_FIX_REPORT.md → docs/reports/
   - MIGRATION_COMPLETE.md → docs/migration/
   - MIGRATION_GUIDE.md → docs/migration/
   - DATA_ARCHITECTURE.txt → docs/architecture/
   - PRODUCTION_README.md → docs/deployment/

4. **Executive & Summary Files:**
   - EXECUTIVE_SUMMARY.py → scripts/reports/

5. **Log Files:**
   - crypto_collector_*.log → logs/collector/
   - data_collector.log → logs/collector/
   - health_check_*.json → logs/monitoring/

## 📁 PROPOSED STRUCTURE
```
crypto-project/
├── app/                          # Application code
├── scripts/                      # Executable scripts
│   ├── analysis/                 # Data analysis scripts
│   ├── data_processing/          # Data cleaning & processing
│   ├── ml/                       # ML training & validation
│   ├── reports/                  # Report generation
│   └── deployment/               # Deployment scripts
├── docs/                         # Documentation
│   ├── reports/                  # Analysis & bug reports
│   ├── migration/                # Migration documentation
│   ├── architecture/             # Architecture docs
│   └── deployment/               # Deployment guides
├── tests/                        # Test files
│   ├── ml/                       # ML model tests
│   └── integration/              # Integration tests
├── demos/                        # Demo files
│   ├── ml/                       # ML demos
│   └── api/                      # API demos
├── logs/                         # Log files
│   ├── collector/                # Data collection logs
│   ├── monitoring/               # Health check logs
│   └── application/              # Application logs
├── data/                         # Data storage
├── config/                       # Configuration
└── [root files]                  # Only essential root files
```

## 🎯 REORGANIZATION STEPS
1. Create missing directory structure
2. Move files to appropriate locations
3. Update import paths in code
4. Update configuration files
5. Test all scripts after reorganization