# 🎯 ML EXAMPLES README

Comprehensive collection of machine learning examples for crypto prediction using the project's ML framework.

## 📚 Examples Overview

### 🟢 [basic_usage.py](./basic_usage.py)
**Perfect for beginners** - Simple, clear examples of how to use each ML algorithm.

**What you'll learn:**
- ✅ Basic Linear Regression for price prediction
- ✅ KNN Classification for trend prediction  
- ✅ KNN Regression for price prediction
- ✅ K-Means Clustering for market analysis
- ✅ Model comparison and selection
- ✅ Simple prediction service simulation

**Run it:**
```bash
cd examples/ml
python basic_usage.py
```

### 🟡 [advanced_pipeline.py](./advanced_pipeline.py)
**For intermediate users** - Production-ready ML pipelines with advanced features.

**What you'll learn:**
- ⚙️ Automated model selection and hyperparameter tuning
- 🎯 Ensemble learning strategies
- 📊 Comprehensive performance analysis
- 💾 Model registry and versioning
- 🚀 Production deployment simulation
- 📈 Advanced monitoring and alerting

**Run it:**
```bash
cd examples/ml
python advanced_pipeline.py
```

### 🔴 [production_examples.py](./production_examples.py)
**For advanced users** - Real-world production scenarios and deployment patterns.

**What you'll learn:**
- 🤖 Complete trading bot with ML predictions
- 🔍 Model drift detection and monitoring
- 🚀 Production API development and testing
- 📊 Performance tracking and analysis
- 🛡️ Error handling and health monitoring
- 💰 Risk management for trading applications

**Run it:**
```bash
cd examples/ml
python production_examples.py
```

## 🚀 Quick Start

1. **Start with basics:**
   ```bash
   python basic_usage.py
   ```

2. **Learn advanced techniques:**
   ```bash
   python advanced_pipeline.py
   ```

3. **Explore production patterns:**
   ```bash
   python production_examples.py
   ```

## 🎯 Algorithm Usage Guide

### 📈 Linear Regression
**Best for:** Fast, interpretable price predictions
```python
from app.ml.algorithms import LinearRegressionModel

model = LinearRegressionModel(target_type='price')
results = model.train(datasets)
predictions = model.predict(new_data)
```

### 🎯 KNN Classification  
**Best for:** Trend direction prediction
```python
from app.ml.algorithms import KNNClassifier

classifier = KNNClassifier(n_neighbors=5, auto_tune=True)
results = classifier.train(datasets)
trend_predictions = classifier.predict(new_data)
```

### 📊 KNN Regression
**Best for:** Non-linear price prediction
```python
from app.ml.algorithms import KNNRegressor

regressor = KNNRegressor(target_type='price', auto_tune=True)
results = regressor.train(datasets)
price_predictions = regressor.predict(new_data)
```

### 🎯 K-Means Clustering
**Best for:** Market regime identification
```python
from app.ml.algorithms import KMeansClusteringModel

clustering = KMeansClusteringModel(auto_tune=True)
results = clustering.train(datasets)
market_clusters = clustering.predict(new_data)
```

## 📊 Example Outputs

### Basic Usage Example Output:
```
🎯 LINEAR REGRESSION - PRICE PREDICTION
✅ Training completed!
   📈 R² Score: 0.9999
   💰 MAE: $23.43

🎯 KNN CLASSIFIER - TREND PREDICTION  
✅ Training completed!
   🎯 Accuracy: 0.835
   📊 Precision: 0.842

🔮 Making predictions...
   Sample 1: Predicted=$45,123.45, Actual=$45,089.12, Error=$34.33
```

### Advanced Pipeline Output:
```
🚀 ADVANCED ML PIPELINE - COMPREHENSIVE ANALYSIS
📊 Step 1: Data Loading and Validation
✅ Loaded datasets: ml_datasets_top3
   📊 Training samples: 2000
   📊 Test samples: 500
   
🤖 Step 2: Automated Model Selection
✅ Tested 3 models
   🏆 Best Regression: Linear Regression (R² = 0.9999)
```

### Production Example Output:
```
🤖 TRADING BOT SIMULATION
🚀 Starting trading simulation with $10000
📈 Trading Period 1:
   Current Price: $45,123.45
   Predicted Price: $45,287.12 (+0.36%)
   Signal: BUY - Strong bullish signal: 5.23% price increase expected
   ✅ Trade executed: BUY 0.044123 BTC at $45,123.45

📊 FINAL RESULTS:
Initial Balance: $10,000.00
Final Balance: $10,847.32
ROI: +8.47%
Win Rate: 65.0%
```

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors:**
   ```bash
   # Make sure you're in the project root
   cd /path/to/crypto-project
   python examples/ml/basic_usage.py
   ```

2. **Data Loading Issues:**
   ```python
   # Ensure datasets exist
   from app.ml.data_prep import load_prepared_datasets
   datasets = load_prepared_datasets('ml_datasets_top3')
   ```

3. **Model Training Errors:**
   ```python
   # Check data shapes and types
   print(f"X_train shape: {datasets['X_train'].shape}")
   print(f"y_train shape: {datasets['y_train'].shape}")
   ```

### Performance Tips:

1. **For faster training:**
   ```python
   # Disable auto-tuning for quick tests
   model = KNNRegressor(auto_tune=False, n_neighbors=5)
   ```

2. **For better accuracy:**
   ```python
   # Enable auto-tuning and use more data
   model = KNNRegressor(auto_tune=True, max_neighbors=15)
   ```

3. **For production use:**
   ```python
   # Save trained models
   from app.ml.model_registry import ModelRegistry
   registry = ModelRegistry()
   registry.save_model(model, 'my_model', 'v1.0')
   ```

## 🎓 Learning Path

1. **Beginner Path:**
   - Start with `basic_usage.py`
   - Read algorithm documentation in `docs/ml/algorithms/`
   - Experiment with different parameters

2. **Intermediate Path:**
   - Study `advanced_pipeline.py`
   - Learn about ensemble methods
   - Understand hyperparameter tuning

3. **Advanced Path:**
   - Explore `production_examples.py`
   - Build your own trading strategies
   - Implement real-time monitoring

## 🔗 Related Documentation

- 📚 [ML Architecture Guide](../../docs/ml/ML_ARCHITECTURE_COMPLETE_GUIDE.md)
- 🔍 [Algorithm Explanations](../../docs/ml/algorithms/)
- 📊 [Performance Analysis](../../docs/ml/ML_ARCHITECTURE_ANALYSIS.md)
- 🚀 [Production Deployment](../../docs/ml/PRODUCTION_DEPLOYMENT.md)

## 💡 Tips for Success

1. **Start Simple:** Begin with basic examples before moving to advanced
2. **Understand the Data:** Know what each feature represents
3. **Validate Results:** Always check predictions against actual values
4. **Monitor Performance:** Track model accuracy over time
5. **Risk Management:** Never risk more than you can afford to lose

## 🤝 Contributing

Found an issue or want to add an example?

1. Create a new example file following the existing pattern
2. Add comprehensive documentation and comments
3. Include error handling and user-friendly output
4. Test thoroughly with different data scenarios

## 📞 Support

If you have questions about the examples:

1. Check the algorithm documentation in `docs/ml/algorithms/`
2. Review the complete ML architecture guide
3. Look at the troubleshooting section above
4. Experiment with the code and observe the outputs

---

**Remember:** These examples are for educational purposes. Always validate thoroughly before using in production trading environments! 🚀