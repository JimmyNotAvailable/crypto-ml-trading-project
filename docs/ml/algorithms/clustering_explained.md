# 🎯 K-MEANS CLUSTERING ALGORITHM EXPLAINED

## 🎯 Algorithm Overview

K-Means là thuật toán unsupervised learning phân cụm dữ liệu thành k clusters dựa trên similarity. Trong project, được sử dụng để phân tích market patterns và market segmentation.

## 🧮 Mathematical Foundation

### Core Objective
```
Minimize: J = Σ(i=1 to k) Σ(x∈Ci) ||x - μi||²

Where:
- J: Within-cluster sum of squares (WCSS)
- k: Number of clusters
- Ci: ith cluster
- μi: Centroid of cluster i
- x: Data point
```

### Algorithm Steps
```
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as cluster means
4. Repeat steps 2-3 until convergence

Convergence: ||μi(t+1) - μi(t)|| < ε
```

### Distance Metrics
```
Euclidean Distance (default):
d(x, μ) = √(Σ(xi - μi)²)

Manhattan Distance:
d(x, μ) = Σ|xi - μi|

Cosine Distance:
d(x, μ) = 1 - (x·μ)/(||x|| ||μ||)
```

## 🛠️ Implementation trong Project

### Class Structure
```python
class KMeansClusteringModel(BaseModel):
    def __init__(self, n_clusters=None, auto_tune=True, max_clusters=10):
        super().__init__('kmeans_clustering', 'clustering')
        self.n_clusters = n_clusters
        self.auto_tune = auto_tune
        self.max_clusters = max_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
```

### Feature Preparation
```python
def _prepare_features(self, datasets):
    """Prepare features for clustering"""
    clustering_features = [
        'open', 'high', 'low', 'close',    # Price features
        'volume',                           # Volume
        'ma_10', 'ma_50',                  # Moving averages
        'volatility',                       # Volatility
        'returns'                          # Returns
    ]
    
    # Remove time-based features that don't make sense for clustering
    X = datasets['train'][clustering_features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X
```

### Optimal Cluster Discovery
```python
def _find_optimal_clusters(self, X_scaled):
    """Find optimal number of clusters using Elbow method and Silhouette score"""
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    
    k_range = range(2, self.max_clusters + 1)
    
    for k in k_range:
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # WCSS (Elbow method)
        wcss.append(kmeans.inertia_)
        
        # Silhouette score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find elbow point
    optimal_k_elbow = self._find_elbow_point(wcss)
    
    # Best silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    # Choose based on silhouette score primarily
    optimal_k = optimal_k_silhouette
    
    return optimal_k, wcss, silhouette_scores
```

### Training Process
```python
def train(self, datasets):
    # 1. Prepare features
    X = self._prepare_features(datasets)
    
    # 2. Feature scaling
    X_scaled = self.scaler.fit_transform(X)
    
    # 3. Find optimal clusters (if not specified)
    if self.auto_tune and self.n_clusters is None:
        optimal_k, wcss, silhouette_scores = self._find_optimal_clusters(X_scaled)
        self.n_clusters = optimal_k
    
    # 4. Train final model
    self.model = KMeans(
        n_clusters=self.n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = self.model.fit_predict(X_scaled)
    
    # 5. Analyze clusters
    cluster_analysis = self._analyze_clusters(X, cluster_labels)
    
    # 6. PCA for visualization
    self.pca = PCA(n_components=2)
    X_pca = self.pca.fit_transform(X_scaled)
    
    return {
        'n_clusters': self.n_clusters,
        'cluster_analysis': cluster_analysis,
        'silhouette_score': silhouette_score(X_scaled, cluster_labels),
        'calinski_harabasz_score': calinski_harabasz_score(X_scaled, cluster_labels),
        'wcss': self.model.inertia_
    }
```

## 📊 Cluster Analysis & Interpretation

### Market Pattern Identification
```python
def _analyze_clusters(self, X, labels):
    """Analyze cluster characteristics"""
    cluster_stats = {}
    
    for cluster_id in range(self.n_clusters):
        cluster_mask = labels == cluster_id
        cluster_data = X[cluster_mask]
        
        stats = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(X) * 100,
            'mean_values': cluster_data.mean().to_dict(),
            'std_values': cluster_data.std().to_dict(),
            'market_interpretation': self._interpret_cluster(cluster_data.mean())
        }
        
        cluster_stats[f'cluster_{cluster_id}'] = stats
    
    return cluster_stats

def _interpret_cluster(self, cluster_mean):
    """Interpret cluster as market condition"""
    # High volatility + high volume = Active trading
    if cluster_mean['volatility'] > 0.05 and cluster_mean['volume'] > cluster_mean['volume'].median():
        return "Active Trading Period"
    
    # Low volatility + low volume = Consolidation
    elif cluster_mean['volatility'] < 0.02 and cluster_mean['volume'] < cluster_mean['volume'].median():
        return "Consolidation Period"
    
    # High returns + high volume = Bull Market
    elif cluster_mean['returns'] > 0.02 and cluster_mean['volume'] > cluster_mean['volume'].median():
        return "Bull Market"
    
    # Negative returns + high volume = Bear Market
    elif cluster_mean['returns'] < -0.02 and cluster_mean['volume'] > cluster_mean['volume'].median():
        return "Bear Market"
    
    else:
        return "Normal Market"
```

## 📈 Performance Metrics

### Clustering Evaluation Metrics
```python
def evaluate(self, X, y=None):
    """Evaluate clustering performance"""
    if not self.is_trained:
        return {}
    
    X_scaled = self.scaler.transform(X)
    labels = self.model.predict(X_scaled)
    
    metrics = {
        'silhouette_score': silhouette_score(X_scaled, labels),
        'calinski_harabasz_score': calinski_harabasz_score(X_scaled, labels),
        'davies_bouldin_score': davies_bouldin_score(X_scaled, labels),
        'inertia': self.model.inertia_,
        'n_clusters': self.n_clusters
    }
    
    return metrics
```

### Typical Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | 0.45-0.65 | Good cluster separation |
| **Calinski-Harabasz** | 500-1500 | Well-defined clusters |
| **Davies-Bouldin** | 0.8-1.2 | Compact clusters |
| **Optimal K** | 4-6 | Market regimes |

## 🎯 Algorithm Strengths & Weaknesses

### ✅ Strengths
1. **Simple**: Easy to understand and implement
2. **Efficient**: O(n×k×i×d) complexity
3. **Guaranteed Convergence**: Always converges
4. **Scalable**: Works with large datasets
5. **Interpretable**: Clear cluster centroids
6. **Memory Efficient**: Stores only centroids

### ❌ Limitations
1. **Requires K**: Need to specify number of clusters
2. **Spherical Clusters**: Assumes clusters are spherical
3. **Sensitive to Initialization**: Random start can affect results
4. **Outliers**: Sensitive to extreme values
5. **Feature Scaling**: Requires normalized features
6. **Equal Sizes**: Tends toward equal-sized clusters

## 🔧 Optimization Techniques

### 1. **Smart Initialization**
```python
# K-means++ initialization (default in sklearn)
kmeans = KMeans(
    n_clusters=k,
    init='k-means++',  # Better initialization
    n_init=10,         # Multiple random starts
    random_state=42
)

# Custom initialization with domain knowledge
def crypto_aware_init(X, k):
    """Initialize centroids based on crypto market knowledge"""
    # Initialize with extreme market conditions
    centroids = []
    
    # High volatility period
    high_vol_idx = X['volatility'].argmax()
    centroids.append(X.iloc[high_vol_idx].values)
    
    # Low volatility period  
    low_vol_idx = X['volatility'].argmin()
    centroids.append(X.iloc[low_vol_idx].values)
    
    # High volume period
    high_vol_idx = X['volume'].argmax()
    centroids.append(X.iloc[high_vol_idx].values)
    
    # Add random centroids for remaining clusters
    remaining = k - len(centroids)
    random_indices = np.random.choice(len(X), remaining, replace=False)
    for idx in random_indices:
        centroids.append(X.iloc[idx].values)
    
    return np.array(centroids)
```

### 2. **Advanced Cluster Selection**
```python
def find_optimal_k_advanced(self, X_scaled):
    """Advanced method combining multiple metrics"""
    k_range = range(2, self.max_clusters + 1)
    
    scores = {
        'wcss': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        scores['wcss'].append(kmeans.inertia_)
        scores['silhouette'].append(silhouette_score(X_scaled, labels))
        scores['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
        scores['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
    
    # Normalize scores and combine
    normalized_scores = {}
    for metric, values in scores.items():
        if metric in ['wcss', 'davies_bouldin']:  # Lower is better
            normalized_scores[metric] = 1 - np.array(values) / max(values)
        else:  # Higher is better
            normalized_scores[metric] = np.array(values) / max(values)
    
    # Combined score
    combined_scores = (
        normalized_scores['silhouette'] * 0.4 +
        normalized_scores['calinski_harabasz'] * 0.3 +
        normalized_scores['wcss'] * 0.2 +
        normalized_scores['davies_bouldin'] * 0.1
    )
    
    optimal_k = k_range[np.argmax(combined_scores)]
    return optimal_k
```

### 3. **Outlier Handling**
```python
def remove_outliers_before_clustering(self, X):
    """Remove outliers using IQR method"""
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter outliers
    mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
    return X[mask]
```

## 🎪 Advanced Features

### 1. **Hierarchical Clustering Comparison**
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def compare_with_hierarchical(self, X_scaled):
    """Compare K-means with hierarchical clustering"""
    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
    hier_labels = hierarchical.fit_predict(X_scaled)
    
    # K-means clustering
    kmeans_labels = self.model.predict(X_scaled)
    
    # Compare results
    from sklearn.metrics import adjusted_rand_score
    similarity = adjusted_rand_score(hier_labels, kmeans_labels)
    
    return {
        'hierarchical_labels': hier_labels,
        'kmeans_labels': kmeans_labels,
        'similarity_score': similarity
    }
```

### 2. **Dynamic Clustering**
```python
class DynamicKMeans:
    """K-means that adapts to changing market conditions"""
    
    def __init__(self, base_k=5, refit_window=1000):
        self.base_k = base_k
        self.refit_window = refit_window
        self.models = []
        self.timestamps = []
    
    def update_model(self, new_data, timestamp):
        """Update clustering model with new data"""
        if len(self.models) == 0 or len(new_data) >= self.refit_window:
            # Train new model
            kmeans = KMeansClusteringModel(n_clusters=self.base_k)
            kmeans.train({'train': new_data})
            
            self.models.append(kmeans)
            self.timestamps.append(timestamp)
            
            # Keep only recent models
            if len(self.models) > 10:
                self.models.pop(0)
                self.timestamps.pop(0)
    
    def predict_current_regime(self, current_data):
        """Predict current market regime"""
        if not self.models:
            return None
        
        latest_model = self.models[-1]
        cluster = latest_model.predict(current_data)
        
        return {
            'cluster': cluster[0],
            'regime': latest_model._interpret_cluster(current_data.mean()),
            'confidence': self._calculate_confidence(current_data, latest_model)
        }
```

### 3. **Cluster Visualization**
```python
def visualize_clusters(self, X, labels):
    """Visualize clusters using PCA"""
    # PCA for 2D visualization
    if self.pca is None:
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(self.scaler.transform(X))
    else:
        X_pca = self.pca.transform(self.scaler.transform(X))
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    # Plot centroids
    centroids_pca = self.pca.transform(self.model.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Crypto Market Clusters (PCA Visualization)')
    plt.colorbar(scatter)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt
```

## 🚀 Production Usage

### Market Regime Detection
```python
def detect_current_market_regime(symbol, lookback_hours=24):
    """Detect current market regime for a symbol"""
    # Get recent data
    data = get_recent_data(symbol, hours=lookback_hours)
    
    # Load trained clustering model
    clustering_model = model_registry.load_model('kmeans_crypto_v2.1')
    
    # Prepare features
    features = prepare_clustering_features(data)
    
    # Predict cluster
    cluster = clustering_model.predict(features.tail(1))[0]
    
    # Get cluster interpretation
    cluster_info = clustering_model.cluster_analysis[f'cluster_{cluster}']
    
    return {
        'symbol': symbol,
        'cluster_id': cluster,
        'market_regime': cluster_info['market_interpretation'],
        'cluster_stats': cluster_info,
        'timestamp': datetime.now()
    }
```

### Portfolio Optimization
```python
def cluster_based_portfolio_optimization(symbols, investment_amount):
    """Optimize portfolio based on market cluster analysis"""
    portfolio = {}
    cluster_allocations = {}
    
    for symbol in symbols:
        regime = detect_current_market_regime(symbol)
        cluster_id = regime['cluster_id']
        
        # Allocate based on market regime
        if regime['market_regime'] == 'Bull Market':
            allocation = 0.4  # Higher allocation to bull markets
        elif regime['market_regime'] == 'Bear Market':
            allocation = 0.1  # Lower allocation to bear markets
        elif regime['market_regime'] == 'Active Trading Period':
            allocation = 0.3  # Medium allocation to volatile periods
        else:
            allocation = 0.2  # Conservative allocation
        
        if cluster_id not in cluster_allocations:
            cluster_allocations[cluster_id] = []
        
        cluster_allocations[cluster_id].append({
            'symbol': symbol,
            'allocation': allocation
        })
    
    # Normalize allocations
    total_allocation = sum(item['allocation'] for cluster in cluster_allocations.values() for item in cluster)
    
    for cluster in cluster_allocations.values():
        for item in cluster:
            normalized_allocation = item['allocation'] / total_allocation
            portfolio[item['symbol']] = {
                'allocation_percent': normalized_allocation * 100,
                'investment_amount': investment_amount * normalized_allocation
            }
    
    return portfolio
```

## 🎯 Use Cases trong Crypto Trading

### 1. **Market Regime Detection**
- Identify bull/bear markets
- Detect consolidation periods
- Recognize high volatility phases

### 2. **Risk Management**
- Cluster-based position sizing
- Diversification across market regimes
- Dynamic stop-loss adjustment

### 3. **Strategy Selection**
- Trend-following in bull clusters
- Mean-reversion in consolidation clusters
- Momentum in high-volatility clusters

### 4. **Anomaly Detection**
- Identify unusual market behavior
- Detect potential market crashes
- Early warning systems

## 🔍 Debugging & Troubleshooting

### Common Issues
1. **Poor Clusters**: Check feature scaling and selection
2. **Empty Clusters**: Reduce k or check initialization
3. **Unstable Results**: Increase n_init parameter
4. **Slow Performance**: Use mini-batch K-means for large data

### Diagnostic Tools
```python
def diagnose_clustering_quality(self, X, labels):
    """Comprehensive clustering quality diagnosis"""
    diagnostics = {}
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    diagnostics['cluster_sizes'] = dict(zip(unique, counts))
    
    # Silhouette analysis per cluster
    silhouette_samples = silhouette_samples_score(X, labels)
    for i in range(self.n_clusters):
        cluster_silhouette = silhouette_samples[labels == i]
        diagnostics[f'cluster_{i}_silhouette'] = {
            'mean': cluster_silhouette.mean(),
            'std': cluster_silhouette.std(),
            'min': cluster_silhouette.min(),
            'max': cluster_silhouette.max()
        }
    
    # Distance to centroids
    distances = []
    for i, centroid in enumerate(self.model.cluster_centers_):
        cluster_points = X[labels == i]
        cluster_distances = np.linalg.norm(cluster_points - centroid, axis=1)
        diagnostics[f'cluster_{i}_distances'] = {
            'mean': cluster_distances.mean(),
            'std': cluster_distances.std(),
            'max': cluster_distances.max()
        }
    
    return diagnostics
```

---

**Kết luận**: K-Means clustering trong project được implement với automatic optimal cluster discovery, comprehensive market interpretation, và advanced visualization để phân tích crypto market patterns và regime detection.