# 🤖 MACHINE LEARNING ARCHITECTURE ANALYSIS

## 📊 Tổng Quan Về Cấu Trúc ML Hiện Tại

Dự án crypto-project sử dụng kiến trúc ML enterprise-grade với 3 thuật toán chính và hệ thống model management hoàn chỉnh.

## 🏗️ Cấu Trúc Thư Mục ML

### 1. **`/app/ml/algorithms/`** - Thuật Toán ML Classes
```
app/ml/algorithms/
├── base.py                  # 🏗️ BaseModel abstract interface
├── linear_regression.py     # 📈 Linear Regression implementation
├── knn_models.py           # 🎯 KNN Classifier & Regressor
├── clustering.py           # 🎯 KMeans Clustering
├── random_forest.py        # 🌳 Random Forest (bonus)
└── __init__.py             # Package exports
```

**Vai trò**: Định nghĩa các thuật toán ML và logic training

### 2. **`/models/`** - Trained Models Storage
```
models/
├── production/             # 🚀 Production models
├── experiments/            # 🧪 Experimental models  
├── staging/                # 🔄 Staging models
├── archived/               # 📦 Archived models
├── backups/                # 💾 Model backups
├── model_registry.json     # 📋 Model metadata registry
├── *.joblib                # 💾 Saved model files
└── *_metadata.json         # 📊 Individual model metadata
```

**Vai trò**: Lưu trữ các models đã train và metadata

## 🎯 Ba Thuật Toán ML Chính

### 1. **📈 Linear Regression** - Dự đoán giá
**File**: `app/ml/algorithms/linear_regression.py`

**Mục đích**:
- Dự đoán giá cryptocurrency (price prediction)
- Dự đoán biến động giá (price_change prediction)

**Cách hoạt động**:
```python
class LinearRegressionModel(BaseModel):
    def __init__(self, target_type='price'):
        # target_type: 'price' hoặc 'price_change'
        
    def train(self, datasets):
        # 1. Chuẩn bị features: open, high, low, close, volume, ma_10, ma_50, volatility, returns, hour
        # 2. Normalization features với StandardScaler
        # 3. Train linear regression model
        # 4. Cross-validation với TimeSeriesSplit
        # 5. Evaluate performance: MAE, RMSE, R²
```

**Features sử dụng**:
- `open`, `high`, `low`, `close` - Giá OHLC
- `volume` - Khối lượng giao dịch
- `ma_10`, `ma_50` - Moving averages
- `volatility` - Độ biến động
- `returns` - Tỷ suất sinh lời
- `hour` - Thời gian trong ngày

**Performance hiện tại**:
- Price prediction: R² = 0.9999, MAE = $23.43
- Price change: R² = 0.9924, MAE = 0.48%

### 2. **🎯 K-Nearest Neighbors (KNN)** - Phân loại & Dự đoán
**File**: `app/ml/algorithms/knn_models.py`

**Hai variants**:

#### **KNNClassifier** - Phân loại xu hướng
```python
class KNNClassifier(BaseModel):
    def train(self, datasets):
        # 1. Tạo trend labels: 'up', 'down', 'stable'
        # 2. Feature scaling với StandardScaler
        # 3. Grid search optimal parameters
        # 4. Train KNeighborsClassifier
        # 5. Evaluate: accuracy, precision, recall, F1
```

**Mục đích**: Phân loại xu hướng giá (tăng/giảm/ổn định)

#### **KNNRegressor** - Dự đoán giá
```python
class KNNRegressor(BaseModel):
    def train(self, datasets):
        # 1. Feature scaling
        # 2. Hyperparameter tuning (n_neighbors, metric, weights)
        # 3. Train KNeighborsRegressor  
        # 4. Evaluate: MAE, RMSE, R²
```

**Mục đích**: Dự đoán giá dựa trên k điểm gần nhất

**Performance**:
- Classifier accuracy: ~83.5%
- Regressor R²: 0.9964, MAE = $271.99

### 3. **🎯 K-Means Clustering** - Phân cụm thị trường
**File**: `app/ml/algorithms/clustering.py`

**Mục đích**:
- Phân cụm các pattern thị trường
- Nhận diện các giai đoạn thị trường khác nhau
- Market segmentation

**Cách hoạt động**:
```python
class KMeansClusteringModel(BaseModel):
    def train(self, datasets):
        # 1. Feature selection và scaling
        # 2. Optimal cluster discovery (Elbow method + Silhouette)
        # 3. Train KMeans với optimal k
        # 4. Cluster analysis và interpretation
        # 5. PCA visualization
```

**Features sử dụng**:
- Price features (open, high, low, close)
- Volume và volatility
- Technical indicators (MA, returns)

**Kết quả**: Phân chia thị trường thành các cụm đặc trưng (bull market, bear market, sideways, etc.)

## 🔄 ML Pipeline Workflow

### 1. **Data Flow**
```
Raw Data → Feature Engineering → Model Training → Evaluation → Production
```

### 2. **Training Process**
```python
# 1. Load datasets
from app.ml.data_prep import load_prepared_datasets
datasets = load_prepared_datasets('ml_datasets_top3')

# 2. Initialize model
from app.ml.algorithms import LinearRegressionModel
model = LinearRegressionModel(target_type='price')

# 3. Train model
metrics = model.train(datasets)

# 4. Auto-register in model registry
model.save_model("linreg_price_v3")
```

### 3. **Model Registry System**
- **File**: `models/model_registry.json`
- **Chức năng**: Track tất cả models, metadata, performance metrics
- **Benefits**: Model versioning, experiment tracking, production deployment

## 🎪 Smart Pipeline System

**File**: `app/ml/smart_pipeline.py`

**Chức năng**:
- Tự động chọn thuật toán tốt nhất cho từng task
- Hyperparameter tuning tự động
- Cross-validation và evaluation
- Model comparison và selection

```python
from app.ml.smart_pipeline import SmartTrainingPipeline

pipeline = SmartTrainingPipeline()
best_model = pipeline.train_with_algorithm_selection(
    datasets=datasets,
    target_type='price',
    algorithms=['linear_regression', 'knn_regressor', 'random_forest']
)
```

## 📊 Model Performance Summary

| Algorithm | Task | R² Score | MAE | Use Case |
|-----------|------|----------|-----|-----------|
| **Linear Regression** | Price Prediction | 0.9999 | $23.43 | Fast, interpretable |
| **Linear Regression** | Price Change | 0.9924 | 0.48% | Trend analysis |
| **KNN Regressor** | Price Prediction | 0.9964 | $271.99 | Non-linear patterns |
| **KNN Classifier** | Trend Classification | 83.5% acc | - | Signal generation |
| **K-Means** | Market Segmentation | - | - | Market analysis |

## 🎯 Algorithm Selection Guidelines

### **Linear Regression** - Khi nào sử dụng?
- ✅ Cần dự đoán nhanh và real-time
- ✅ Muốn hiểu feature importance
- ✅ Data có linear relationship
- ✅ Cần model đơn giản, dễ deploy

### **KNN** - Khi nào sử dụng?
- ✅ Data có non-linear patterns
- ✅ Cần capture local patterns
- ✅ Có nhiều data points
- ✅ Muốn model adaptive

### **K-Means** - Khi nào sử dụng?
- ✅ Market analysis và segmentation
- ✅ Anomaly detection
- ✅ Portfolio optimization
- ✅ Risk assessment

## 🚀 Production Usage

### **Prediction Service**
```python
from app.ml.evaluate import load_trained_models

models = load_trained_models()
price_prediction = models['linear_regression']['price_model'].predict(new_data)
trend_prediction = models['knn']['classifier'].predict(new_data)
```

### **Model Management**
```python
from app.ml.model_registry import model_registry

# Get best model for specific task
best_model = model_registry.get_best_model(target_type='price')

# Deploy to production
model_registry.deploy_model(model_id, environment='production')
```

---

**Kết luận**: Dự án sử dụng kiến trúc ML enterprise-grade với 3 thuật toán bổ trợ lẫn nhau, system hoàn chỉnh cho training, evaluation, và production deployment.