# 📂 PROJECT STRUCTURE EXPLANATION

## 🏗️ TWO DISTINCT "MODELS" DIRECTORIES

### 📁 `/models/` - **MODEL ARTIFACTS & STORAGE**
**Purpose**: Store trained model files and metadata
**Contains**:
- ✅ `.joblib` files (serialized trained models)
- ✅ `.json` metadata files
- ✅ `model_registry.json` (model lifecycle tracking)
- ✅ Organized subdirectories:
  - 📁 `production/` - Live production models
  - 📁 `staging/` - Models ready for deployment
  - 📁 `experiments/` - Research/development models
  - 📁 `archived/` - Old versions
  - 📁 `backups/` - Model backups

**Usage Example**:
```python
# Load a trained model
model = joblib.load("models/production/linreg_price_v2.joblib")
predictions = model.predict(X)
```

---

### 📁 `/app/ml/algorithms/` - **MODEL ALGORITHM CLASSES**
**Purpose**: Define ML algorithm implementations and training logic
**Contains**:
- ✅ `base.py` - BaseModel abstract interface
- ✅ `linear_regression.py` - LinearRegressionModel class
- ✅ `knn_models.py` - KNN classifier/regressor classes
- ✅ `clustering.py` - KMeans clustering class
- ✅ `__init__.py` - Package exports

**Usage Example**:
```python
# Train a new model
from app.ml.algorithms import LinearRegressionModel
model = LinearRegressionModel(target_type='price')
model.train(datasets)
model.save_model("linreg_price_v3")  # Saves to /models/
```

---

## 🔄 ML PIPELINE FLOW

```
1. ALGORITHM DEFINITION (/app/ml/algorithms/)
   ↓
2. TRAINING PROCESS (algorithms create instances)
   ↓
3. MODEL ARTIFACTS (/models/ storage)
   ↓
4. PRODUCTION DEPLOYMENT (load from /models/)
```

---

## 🎯 WHY THIS SEPARATION?

### ✅ **Clear Responsibilities**:
- `/algorithms/` = CODE (how to train)
- `/models/` = DATA (trained results)

### ✅ **Version Control**:
- Code changes tracked in git
- Model artifacts managed separately

### ✅ **Deployment**:
- Deploy algorithms to training servers
- Deploy model artifacts to prediction servers

### ✅ **Collaboration**:
- Data scientists work on `/algorithms/`
- MLOps engineers manage `/models/`

---

## 📋 NAMING CONVENTION

| Directory | Purpose | Example Files |
|-----------|---------|---------------|
| `/models/` | **Artifacts** | `linreg_price.joblib`, `model_registry.json` |
| `/app/ml/algorithms/` | **Code** | `linear_regression.py`, `base.py` |

**This naming avoids confusion and follows enterprise ML best practices! 🚀**