# 🎯 K-NEAREST NEIGHBORS (KNN) ALGORITHM EXPLAINED

## 🎯 Algorithm Overview

KNN là thuật toán lazy learning dự đoán kết quả dựa trên k điểm gần nhất trong không gian feature. Project sử dụng 2 variants: **KNNClassifier** (phân loại xu hướng) và **KNNRegressor** (dự đoán giá).

## 🧮 Mathematical Foundation

### Core Principle
```
Prediction = f(k nearest neighbors)

Distance Metrics:
- Euclidean: d = √(Σ(xᵢ - yᵢ)²)
- Manhattan: d = Σ|xᵢ - yᵢ|
- Minkowski: d = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```

### Classification Rule
```
ŷ = mode(y₁, y₂, ..., yₖ)

Where:
- ŷ: Predicted class
- yᵢ: Class of ith nearest neighbor
- mode: Most frequent class
```

### Regression Rule
```
ŷ = (1/k) * Σwᵢyᵢ

Where:
- wᵢ: Weight of ith neighbor
- wᵢ = 1 (uniform) or wᵢ = 1/dᵢ (distance-weighted)
```

## 🛠️ Implementation trong Project

### 1. **KNNClassifier** - Trend Prediction

#### Class Structure
```python
class KNNClassifier(BaseModel):
    def __init__(self, n_neighbors=5, auto_tune=True):
        self.n_neighbors = n_neighbors
        self.auto_tune = auto_tune
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
```

#### Target Engineering
```python
def _create_trend_labels(self, price_change):
    """Convert price changes to trend categories"""
    conditions = [
        price_change > 0.02,    # Up trend (>2% increase)
        price_change < -0.02,   # Down trend (>2% decrease)
        True                    # Stable trend (else)
    ]
    choices = ['up', 'down', 'stable']
    return np.select(conditions, choices)
```

#### Hyperparameter Tuning
```python
def train(self, datasets):
    if self.auto_tune:
        param_grid = {
            'n_neighbors': [3, 5, 7, 11, 15, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
```

### 2. **KNNRegressor** - Price Prediction

#### Class Structure
```python
class KNNRegressor(BaseModel):
    def __init__(self, target_type='price', n_neighbors=5, auto_tune=True):
        self.target_type = target_type  # 'price' or 'price_change'
        self.n_neighbors = n_neighbors
        self.auto_tune = auto_tune
        self.model = None
        self.scaler = StandardScaler()
```

#### Training Process
```python
def train(self, datasets):
    # 1. Feature preparation & scaling
    X_train_scaled = self.scaler.fit_transform(X_train)
    
    # 2. Hyperparameter tuning
    if self.auto_tune:
        best_params = self._tune_hyperparameters(X_train_scaled, y_train)
        self.model = KNeighborsRegressor(**best_params)
    
    # 3. Model training
    self.model.fit(X_train_scaled, y_train)
    
    # 4. Performance evaluation
    return self._evaluate_performance(X_test, y_test)
```

## 📊 Performance Analysis

### Current Performance
| Model Type | Metric | Value | Use Case |
|------------|--------|-------|----------|
| **KNNClassifier** | Accuracy | 83.5% | Trend prediction |
| **KNNClassifier** | Precision | 83.9% | Signal quality |
| **KNNClassifier** | Recall | 83.5% | Signal coverage |
| **KNNRegressor** | R² Score | 0.9964 | Price prediction |
| **KNNRegressor** | MAE | $271.99 | Average error |
| **KNNRegressor** | RMSE | $698.12 | Error variance |

### Optimal Hyperparameters
```python
# KNNClassifier optimal params
classifier_params = {
    'n_neighbors': 21,
    'weights': 'distance',
    'metric': 'manhattan'
}

# KNNRegressor optimal params  
regressor_params = {
    'n_neighbors': 5,
    'weights': 'distance',
    'metric': 'euclidean'
}
```

## 🎯 Algorithm Strengths & Weaknesses

### ✅ Strengths
1. **No Assumptions**: No linear/parametric assumptions
2. **Non-linear**: Captures complex patterns
3. **Local Learning**: Adapts to local data structure
4. **Multi-class**: Natural multi-class support
5. **Simple**: Easy to understand and implement
6. **Robust**: Works with noisy data

### ❌ Limitations
1. **Computational Cost**: O(n) prediction time
2. **Memory**: Stores entire training set
3. **Curse of Dimensionality**: Poor with many features
4. **Sensitive to Scale**: Requires normalization
5. **Imbalanced Data**: Biased towards majority class
6. **No Model**: No learned parameters

## 🔧 Optimization Techniques

### 1. **Feature Scaling**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling (most common)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Robust scaling (for outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. **Dimensionality Reduction**
```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X_scaled)

# Feature selection
selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)
```

### 3. **Distance Metric Selection**
```python
# Try different distance metrics
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

best_metric = None
best_score = -np.inf

for metric in distance_metrics:
    knn = KNeighborsRegressor(n_neighbors=5, metric=metric)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='r2')
    
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_metric = metric
```

### 4. **Weighted Predictions**
```python
# Distance-weighted predictions
knn = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance'  # Weight by inverse distance
)

# Custom weight function
def custom_weights(distances):
    return 1 / (distances + 1e-8)  # Avoid division by zero

knn = KNeighborsRegressor(
    n_neighbors=5,
    weights=custom_weights
)
```

## 🎪 Advanced Features

### 1. **Ensemble KNN**
```python
class EnsembleKNN:
    def __init__(self, k_values=[3, 5, 7, 11]):
        self.k_values = k_values
        self.models = []
        
    def fit(self, X, y):
        for k in self.k_values:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X, y)
            self.models.append(knn)
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average ensemble
        return np.mean(predictions, axis=0)
```

### 2. **Adaptive KNN**
```python
class AdaptiveKNN:
    def __init__(self, min_k=3, max_k=15):
        self.min_k = min_k
        self.max_k = max_k
        
    def _select_optimal_k(self, X_query, X_train, y_train):
        """Select optimal k for each query point"""
        optimal_k = []
        
        for query in X_query:
            best_k = self.min_k
            best_error = np.inf
            
            for k in range(self.min_k, self.max_k + 1):
                knn = KNeighborsRegressor(n_neighbors=k)
                # Cross-validation for this k
                errors = cross_val_score(knn, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
                
                if -errors.mean() < best_error:
                    best_error = -errors.mean()
                    best_k = k
                    
            optimal_k.append(best_k)
        
        return optimal_k
```

### 3. **Local Outlier Detection**
```python
from sklearn.neighbors import LocalOutlierFactor

def remove_outliers_with_lof(X, contamination=0.1):
    """Remove outliers using Local Outlier Factor"""
    lof = LocalOutlierFactor(contamination=contamination)
    outlier_labels = lof.fit_predict(X)
    
    # Keep only inliers (label = 1)
    return X[outlier_labels == 1]
```

## 🚀 Production Usage

### Real-time Trend Prediction
```python
def predict_trend(symbol, current_data):
    # Load trained classifier
    classifier = model_registry.load_model('knn_classifier_v2.1')
    
    # Prepare features
    features = prepare_features(current_data)
    features_scaled = classifier.scaler.transform(features)
    
    # Predict trend
    trend = classifier.predict(features_scaled)[0]
    confidence = classifier.predict_proba(features_scaled)[0].max()
    
    return {
        'symbol': symbol,
        'trend': trend,  # 'up', 'down', 'stable'
        'confidence': confidence,
        'timestamp': datetime.now()
    }
```

### Batch Price Prediction
```python
def batch_predict_prices(symbols, timeframe='1h'):
    predictions = {}
    
    # Load trained regressor
    regressor = model_registry.load_model('knn_regressor_v2.1')
    
    for symbol in symbols:
        data = get_latest_data(symbol, timeframe)
        features = prepare_features(data)
        features_scaled = regressor.scaler.transform(features)
        
        predicted_price = regressor.predict(features_scaled)[0]
        
        predictions[symbol] = {
            'current_price': data['close'].iloc[-1],
            'predicted_price': predicted_price,
            'change_percent': ((predicted_price - data['close'].iloc[-1]) / data['close'].iloc[-1]) * 100
        }
    
    return predictions
```

## 🎯 Algorithm Variants in Project

### 1. **Classification Variants**
```python
# Trend classification (3 classes)
trends = ['up', 'down', 'stable']

# Volatility classification (5 classes)  
volatility_levels = ['very_low', 'low', 'medium', 'high', 'very_high']

# Volume classification (3 classes)
volume_levels = ['low_volume', 'normal_volume', 'high_volume']
```

### 2. **Regression Variants**
```python
# Price prediction
target_type = 'price'

# Price change prediction
target_type = 'price_change'

# Return prediction
target_type = 'returns'
```

## 🔍 Model Interpretation

### Feature Importance via Permutation
```python
from sklearn.inspection import permutation_importance

def get_feature_importance(model, X_test, y_test):
    # Permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=42
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    })
    
    return importance_df.sort_values('importance', ascending=False)
```

### Neighbor Analysis
```python
def analyze_neighbors(model, query_point, X_train, y_train):
    """Analyze the neighbors for a prediction"""
    distances, indices = model.kneighbors(query_point.reshape(1, -1))
    
    neighbors_info = []
    for i, idx in enumerate(indices[0]):
        neighbors_info.append({
            'distance': distances[0][i],
            'target_value': y_train.iloc[idx],
            'features': X_train.iloc[idx].to_dict()
        })
    
    return neighbors_info
```

## 🎯 When to Use KNN

### ✅ **KNNClassifier Ideal For**
- Trend/signal classification
- Non-linear decision boundaries
- Local pattern recognition
- Multi-class problems
- When you need probability estimates

### ✅ **KNNRegressor Ideal For**
- Non-linear price patterns
- Local market behavior
- Short-term predictions
- When linear models fail
- Capturing market regimes

### ❌ **Not Suitable When**
- Real-time high-frequency trading
- Very large datasets (>100k points)
- High-dimensional features (>50)
- Need interpretable models
- Limited computational resources

## 🔍 Debugging & Troubleshooting

### Common Issues
1. **Poor Performance**: Check feature scaling
2. **Slow Predictions**: Reduce k or use approximate methods
3. **Overfitting**: Increase k value
4. **Underfitting**: Decrease k value

### Diagnostic Tools
```python
def plot_k_performance(X, y, k_range=range(1, 31)):
    """Plot performance vs k value"""
    train_scores = []
    val_scores = []
    
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        
        # Training score
        knn.fit(X, y)
        train_score = knn.score(X, y)
        train_scores.append(train_score)
        
        # Validation score
        val_score = cross_val_score(knn, X, y, cv=5).mean()
        val_scores.append(val_score)
    
    plt.plot(k_range, train_scores, label='Training')
    plt.plot(k_range, val_scores, label='Validation')
    plt.xlabel('k value')
    plt.ylabel('R² Score')
    plt.legend()
    plt.title('KNN Performance vs k')
```

---

**Kết luận**: KNN trong project được implement với automatic hyperparameter tuning, comprehensive evaluation, và optimized cho crypto market patterns với high accuracy cho cả classification và regression tasks.