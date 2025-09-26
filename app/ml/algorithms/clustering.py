"""
🎯 CLUSTERING MODELS
==================

Enterprise-grade KMeans clustering for crypto market analysis:
- ✅ Market pattern segmentation
- ✅ Automatic optimal cluster discovery
- ✅ Feature engineering & preprocessing
- ✅ Clustering evaluation metrics
- ✅ Visualization & analysis tools
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .base import BaseModel

class KMeansClusteringModel(BaseModel):
    """
    🎯 KMeans Clustering for Crypto Market Segmentation
    
    Features:
    - Market pattern discovery
    - Automatic cluster optimization
    - Comprehensive evaluation
    - Visualization tools
    """
    
    def __init__(self, n_clusters: Optional[int] = None, auto_tune: bool = True, max_clusters: int = 10):
        """
        Initialize KMeans Clustering model
        
        Args:
            n_clusters: Number of clusters (if auto_tune=False)
            auto_tune: Whether to automatically find optimal clusters
            max_clusters: Maximum clusters to test when auto_tune=True
        """
        super().__init__(
            model_name="kmeans_clustering",
            model_type="clustering"
        )
        
        self.n_clusters = n_clusters
        self.auto_tune = auto_tune
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.cluster_centers_ = None
        self.labels_ = None
        
        # Update metadata
        self.metadata['n_clusters'] = str(n_clusters) if n_clusters else "auto"
        self.metadata['auto_tune'] = str(auto_tune)
        self.metadata['max_clusters'] = str(max_clusters)
        self.metadata['algorithm'] = 'KMeans'
    
    def _prepare_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        🔧 Prepare features for clustering
        
        Args:
            datasets: Dict containing train/test data
            
        Returns:
            Feature dataframe ready for clustering
        """
        print("🔧 Preparing features for clustering...")
        
        # Get training data
        train_data = datasets['train'].copy()
        
        # Define feature columns for clustering
        # Focus on technical indicators and patterns
        exclude_cols = ['date', 'symbol', 'target_price', 'target_price_change', 'target_trend']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("❌ No feature columns found for clustering")
        
        self.feature_columns = feature_cols
        X = train_data[feature_cols].copy()
        
        # Remove rows with NaN values
        X = X.dropna()
        
        print(f"✅ Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Features: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
        
        return X
    
    def _find_optimal_clusters(self, X_scaled: np.ndarray) -> int:
        """
        🔧 Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Optimal number of clusters
        """
        print("🔧 Finding optimal number of clusters...")
        
        # Test different cluster numbers
        cluster_range = range(2, min(self.max_clusters + 1, len(X_scaled) // 10))
        inertias = []
        silhouette_scores = []
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Find elbow point (approximate)
        # Use second derivative to find elbow
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_derivatives.append(inertias[i-1] - 2*inertias[i] + inertias[i+1])
            
            # Find the point with maximum second derivative (sharpest elbow)
            elbow_idx = np.argmax(second_derivatives) + 1  # +1 because we start from index 1
            elbow_clusters = list(cluster_range)[elbow_idx]
        else:
            elbow_clusters = 3  # Default fallback
        
        # Find best silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_silhouette_clusters = list(cluster_range)[best_silhouette_idx]
        
        # Choose based on silhouette score if it's reasonable, otherwise use elbow
        if silhouette_scores[best_silhouette_idx] > 0.3:
            optimal_clusters = best_silhouette_clusters
            selection_method = "silhouette"
        else:
            optimal_clusters = elbow_clusters
            selection_method = "elbow"
        
        print(f"✅ Optimal clusters: {optimal_clusters} (method: {selection_method})")
        print(f"📊 Silhouette score: {silhouette_scores[best_silhouette_idx]:.3f}")
        
        # Store evaluation metrics
        self.cluster_evaluation = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_clusters': optimal_clusters,
            'selection_method': selection_method
        }
        
        return optimal_clusters
    
    def train(self, datasets: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        🎯 Train KMeans clustering model
        
        Args:
            datasets: Dict containing 'train' and 'test' dataframes
            **kwargs: Additional training parameters
            
        Returns:
            Dict with clustering metrics and results
        """
        print("🎯 Training KMeans clustering...")
        
        # Prepare data
        X = self._prepare_features(datasets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal clusters if auto_tune is enabled
        if self.auto_tune:
            optimal_clusters = self._find_optimal_clusters(X_scaled)
            self.n_clusters = optimal_clusters
        elif self.n_clusters is None:
            self.n_clusters = min(5, len(X) // 10)  # Default heuristic
        
        # Train final model
        print(f"🎯 Training with {self.n_clusters} clusters...")
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        self.model.fit(X_scaled)
        
        # Store results
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(X_scaled, self.labels_)
        calinski_harabasz = calinski_harabasz_score(X_scaled, self.labels_)
        inertia = self.model.inertia_
        
        # Create cluster analysis
        cluster_analysis = self._analyze_clusters(X, self.labels_)
        
        # Prepare metrics
        train_metrics = {
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'inertia': inertia,
            'cluster_sizes': cluster_analysis['cluster_sizes'],
            'cluster_stats': cluster_analysis['cluster_stats']
        }
        
        # Add optimization results if available
        if hasattr(self, 'cluster_evaluation'):
            train_metrics['optimization_results'] = self.cluster_evaluation
        
        # Update model state
        self.is_trained = True
        feature_count = len(self.feature_columns) if self.feature_columns else 0
        self.training_history = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_features': feature_count,
            'n_samples': len(X),
            'metrics': train_metrics
        }
        
        # Update metadata
        feature_count = len(self.feature_columns) if self.feature_columns else 0
        self.metadata['training_samples'] = str(len(X))
        self.metadata['feature_count'] = str(feature_count)
        self.metadata['final_n_clusters'] = str(self.n_clusters) if self.n_clusters else "unknown"
        self.metadata['last_trained'] = pd.Timestamp.now().isoformat()
        
        # Print results
        print(f"✅ Clustering completed!")
        print(f"📊 Clusters: {self.n_clusters}")
        print(f"📊 Silhouette Score: {silhouette_avg:.3f}")
        print(f"📊 Cluster sizes: {cluster_analysis['cluster_sizes']}")
        
        return train_metrics
    
    def _analyze_clusters(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        📊 Analyze cluster characteristics
        
        Args:
            X: Original feature data
            labels: Cluster labels
            
        Returns:
            Dict with cluster analysis results
        """
        cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        
        # Calculate cluster statistics for each feature
        cluster_stats = {}
        X_with_clusters = X.copy()
        X_with_clusters['cluster'] = labels
        
        n_clusters = self.n_clusters or 3  # Fallback to 3 if None
        for cluster_id in range(n_clusters):
            cluster_data = X_with_clusters[X_with_clusters['cluster'] == cluster_id]
            cluster_stats[cluster_id] = {
                'size': len(cluster_data),
                'mean_values': cluster_data.drop('cluster', axis=1).mean().to_dict(),
                'std_values': cluster_data.drop('cluster', axis=1).std().to_dict()
            }
        
        return {
            'cluster_sizes': cluster_sizes,
            'cluster_stats': cluster_stats
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        🔮 Predict cluster assignments
        
        Args:
            X: Feature dataframe
            
        Returns:
            Cluster assignments
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained before making predictions")
        
        # Ensure we have the right features
        if self.feature_columns:
            X = X[self.feature_columns].copy()
        
        # Remove NaN values
        X = X.dropna()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        📊 Evaluate clustering performance
        
        Args:
            X: Test features
            y: Not used for clustering (unsupervised)
            
        Returns:
            Dict with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained before evaluation")
        
        # Get cluster assignments
        cluster_labels = self.predict(X)
        
        # Prepare scaled data for metrics
        if self.feature_columns:
            X = X[self.feature_columns].copy()
        X = X.dropna()
        X_scaled = self.scaler.transform(X)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        
        return {
            'silhouette_score': float(silhouette_avg),
            'calinski_harabasz_score': float(calinski_harabasz),
            'n_clusters': len(np.unique(cluster_labels))
        }
    
    def get_cluster_centers(self) -> pd.DataFrame:
        """
        📊 Get cluster centers in original feature space
        
        Returns:
            DataFrame with cluster centers
        """
        if not self.is_trained or self.cluster_centers_ is None:
            raise ValueError("❌ Model must be trained to get cluster centers")
        
        # Inverse transform cluster centers to original scale
        centers_original = self.scaler.inverse_transform(self.cluster_centers_)
        
        n_clusters = self.n_clusters or 3  # Fallback
        centers_df = pd.DataFrame(
            centers_original,
            columns=self.feature_columns,
            index=[f'Cluster_{i}' for i in range(n_clusters)]
        )
        
        return centers_df
    
    def plot_clusters_2d(self, X: pd.DataFrame, use_pca: bool = True) -> mfig.Figure:
        """
        📊 Plot clusters in 2D space
        
        Args:
            X: Feature data used for clustering
            use_pca: Whether to use PCA for dimensionality reduction
            
        Returns:
            Matplotlib figure
        """
        if not self.is_trained:
            raise ValueError("❌ Model must be trained to plot clusters")
        
        # Prepare data
        if self.feature_columns:
            X_plot = X[self.feature_columns].copy()
        else:
            X_plot = X.copy()
        
        X_plot = X_plot.dropna()
        X_scaled = self.scaler.transform(X_plot)
        
        # Get cluster labels
        labels = self.model.predict(X_scaled)
        
        # Reduce to 2D
        if use_pca or X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
            if self.cluster_centers_ is not None:
                centers_2d = pca.transform(self.cluster_centers_)
            else:
                centers_2d = None
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
        else:
            X_2d = X_scaled[:, :2]
            if self.cluster_centers_ is not None:
                centers_2d = self.cluster_centers_[:, :2]
            else:
                centers_2d = None
            xlabel = self.feature_columns[0] if self.feature_columns else 'Feature 1'
            ylabel = (self.feature_columns[1] if self.feature_columns and len(self.feature_columns) > 1 
                     else 'Feature 2')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot points
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels,
            cmap='Set3',
            alpha=0.7,
            s=50
        )
        
        # Plot cluster centers if available
        if centers_2d is not None:
            ax.scatter(
                centers_2d[:, 0], centers_2d[:, 1],
                c='red',
                marker='x',
                s=200,
                linewidths=3,
                label='Centroids'
            )
        
        # Customize plot
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'KMeans Clustering Results\\n{self.n_clusters} Clusters, Silhouette Score: {silhouette_score(X_scaled, labels):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_evaluation(self) -> mfig.Figure:
        """
        📊 Plot cluster evaluation metrics
        
        Returns:
            Matplotlib figure with elbow curve and silhouette scores
        """
        if not hasattr(self, 'cluster_evaluation'):
            raise ValueError("❌ Cluster evaluation data not available. Run training with auto_tune=True.")
        
        eval_data = self.cluster_evaluation
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow curve
        ax1.plot(eval_data['cluster_range'], eval_data['inertias'], 'bo-', linewidth=2)
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Inertia (WCSS)')
        ax1.set_title('Elbow Method for Optimal Clusters')
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal point
        optimal_k = eval_data['optimal_clusters']
        optimal_idx = eval_data['cluster_range'].index(optimal_k)
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_k}')
        ax1.legend()
        
        # Silhouette scores
        ax2.plot(eval_data['cluster_range'], eval_data['silhouette_scores'], 'go-', linewidth=2)
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        # Mark optimal point
        ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_k}')
        ax2.legend()
        
        plt.tight_layout()
        return fig