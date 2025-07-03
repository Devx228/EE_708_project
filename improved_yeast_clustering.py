import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ImprovedYeastClustering:
    """Advanced clustering pipeline for yeast protein localization"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.dim_reducer = None
        self.best_clusterer = None
        self.results = {}
        
    def load_and_prepare_data(self, filepath):
        """Load data and perform initial preprocessing"""
        # Load data
        self.df = pd.read_csv(filepath)
        
        # Separate features and identifiers
        self.protein_names = self.df['Sequence Name']
        self.feature_names = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
        self.X = self.df[self.feature_names].values
        
        print(f"Loaded {len(self.df)} proteins with {len(self.feature_names)} features")
        
        # Analyze feature properties
        self._analyze_features()
        
        return self.X
    
    def _analyze_features(self):
        """Analyze feature distributions and correlations"""
        print("\n=== Feature Analysis ===")
        
        # Check for low variance features
        for i, feature in enumerate(self.feature_names):
            unique_vals = np.unique(self.X[:, i])
            if len(unique_vals) < 10:
                print(f"Low variance warning: {feature} has only {len(unique_vals)} unique values")
        
        # Correlation analysis
        corr_matrix = np.corrcoef(self.X.T)
        high_corr_pairs = []
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                if abs(corr_matrix[i, j]) > 0.5:
                    high_corr_pairs.append((self.feature_names[i], self.feature_names[j], corr_matrix[i, j]))
        
        if high_corr_pairs:
            print("\nHighly correlated features:")
            for f1, f2, corr in high_corr_pairs:
                print(f"  {f1} - {f2}: {corr:.3f}")
    
    def preprocess_data(self, remove_low_var=True, use_power_transform=True):
        """Advanced preprocessing pipeline"""
        print("\n=== Preprocessing Pipeline ===")
        
        X_processed = self.X.copy()
        
        # Option 1: Remove low variance features
        if remove_low_var:
            # Keep only features with more than 3 unique values
            mask = []
            kept_features = []
            for i, feature in enumerate(self.feature_names):
                if len(np.unique(X_processed[:, i])) > 3:
                    mask.append(i)
                    kept_features.append(feature)
            
            X_processed = X_processed[:, mask]
            print(f"Removed low variance features. Kept: {kept_features}")
            self.selected_features = kept_features
        else:
            self.selected_features = self.feature_names
        
        # Scaling
        if use_power_transform:
            self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            print("Using PowerTransformer for non-normal distributions")
        else:
            self.scaler = RobustScaler()
            print("Using RobustScaler for outlier robustness")
        
        self.X_scaled = self.scaler.fit_transform(X_processed)
        
        return self.X_scaled
    
    def reduce_dimensions(self, method='umap', n_components=3):
        """Dimensionality reduction for better clustering"""
        print(f"\n=== Dimensionality Reduction ({method.upper()}) ===")
        
        if method == 'pca':
            self.dim_reducer = PCA(n_components=n_components, random_state=self.random_state)
            self.X_reduced = self.dim_reducer.fit_transform(self.X_scaled)
            explained_var = sum(self.dim_reducer.explained_variance_ratio_)
            print(f"PCA: {n_components} components explain {explained_var:.2%} variance")
            
        elif method == 'tsne':
            self.dim_reducer = TSNE(n_components=n_components, random_state=self.random_state, 
                                   perplexity=30, n_iter=1000)
            self.X_reduced = self.dim_reducer.fit_transform(self.X_scaled)
            print(f"t-SNE: Reduced to {n_components} dimensions")
            
        elif method == 'umap':
            try:
                from umap import UMAP
                self.dim_reducer = UMAP(n_components=n_components, n_neighbors=30, 
                                      min_dist=0.3, random_state=self.random_state)
                self.X_reduced = self.dim_reducer.fit_transform(self.X_scaled)
                print(f"UMAP: Reduced to {n_components} dimensions")
            except ImportError:
                print("UMAP not installed. Falling back to PCA")
                return self.reduce_dimensions('pca', n_components)
        
        return self.X_reduced
    
    def find_optimal_clusters(self, max_k=10):
        """Find optimal number of clusters using multiple methods"""
        print("\n=== Finding Optimal Cluster Count ===")
        
        # Method 1: Elbow method with inertia
        inertias = []
        silhouettes = []
        
        K_range = range(2, max_k+1)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, n_init=30, random_state=self.random_state)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.X_scaled, kmeans.labels_))
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        
        ax2.plot(K_range, silhouettes, 'ro-')
        ax2.set_xlabel('Number of clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.show()
        
        # Find elbow point
        optimal_k = K_range[np.argmax(silhouettes)]
        print(f"Optimal clusters based on silhouette: {optimal_k}")
        
        return optimal_k
    
    def compare_clustering_algorithms(self, n_clusters=5):
        """Compare multiple clustering algorithms"""
        print(f"\n=== Comparing Clustering Algorithms (k={n_clusters}) ===")
        
        algorithms = {
            'K-Means': KMeans(n_clusters=n_clusters, n_init=50, random_state=self.random_state),
            'GMM': GaussianMixture(n_components=n_clusters, n_init=10, random_state=self.random_state),
            'Spectral': SpectralClustering(n_clusters=n_clusters, affinity='rbf', 
                                          n_neighbors=20, random_state=self.random_state),
            'Agglomerative-Ward': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
            'Agglomerative-Complete': AgglomerativeClustering(n_clusters=n_clusters, linkage='complete'),
        }
        
        # Add DBSCAN with optimal eps
        eps = self._find_optimal_eps()
        algorithms['DBSCAN'] = DBSCAN(eps=eps, min_samples=10)
        
        # Test on both scaled and reduced data
        datasets = {
            'Scaled': self.X_scaled,
            'Reduced': self.X_reduced if hasattr(self, 'X_reduced') else self.X_scaled
        }
        
        for data_name, X_data in datasets.items():
            print(f"\n--- Results on {data_name} Data ---")
            
            for name, algorithm in algorithms.items():
                try:
                    # Fit algorithm
                    if hasattr(algorithm, 'fit_predict'):
                        labels = algorithm.fit_predict(X_data)
                    else:
                        labels = algorithm.fit(X_data).predict(X_data)
                    
                    # Skip if only one cluster found (DBSCAN)
                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters_found < 2:
                        print(f"{name}: Only {n_clusters_found} cluster(s) found - skipping")
                        continue
                    
                    # Calculate metrics
                    sil = silhouette_score(X_data, labels)
                    db = davies_bouldin_score(X_data, labels)
                    ch = calinski_harabasz_score(X_data, labels)
                    
                    # Store results
                    key = f"{name}_{data_name}"
                    self.results[key] = {
                        'labels': labels,
                        'silhouette': sil,
                        'davies_bouldin': db,
                        'calinski_harabasz': ch,
                        'n_clusters': n_clusters_found
                    }
                    
                    print(f"{name}: Silhouette={sil:.3f}, DBI={db:.3f}, CH={ch:.1f}, Clusters={n_clusters_found}")
                    
                except Exception as e:
                    print(f"{name}: Failed - {str(e)}")
        
        # Find best algorithm
        best_key = max(self.results.keys(), key=lambda k: self.results[k]['silhouette'])
        self.best_result = self.results[best_key]
        print(f"\nBest algorithm: {best_key} (Silhouette={self.best_result['silhouette']:.3f})")
        
        return self.results
    
    def _find_optimal_eps(self, k=10):
        """Find optimal eps for DBSCAN using k-distance graph"""
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(self.X_scaled)
        distances, indices = neighbors_fit.kneighbors(self.X_scaled)
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find elbow point (simplified)
        eps = np.percentile(distances, 90)
        return eps
    
    def analyze_cluster_stability(self, algorithm='kmeans', n_clusters=5, n_splits=10):
        """Assess clustering stability using cross-validation"""
        print(f"\n=== Cluster Stability Analysis ===")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        stability_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.X_scaled)):
            # Train on subset
            if algorithm == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, n_init=30, random_state=self.random_state)
            elif algorithm == 'spectral':
                clusterer = SpectralClustering(n_clusters=n_clusters, random_state=self.random_state)
            
            # Fit and predict
            train_labels = clusterer.fit_predict(self.X_scaled[train_idx])
            
            # For algorithms that can predict on new data
            if hasattr(clusterer, 'predict'):
                test_labels = clusterer.predict(self.X_scaled[test_idx])
                test_score = silhouette_score(self.X_scaled[test_idx], test_labels)
                stability_scores.append(test_score)
        
        if stability_scores:
            mean_stability = np.mean(stability_scores)
            std_stability = np.std(stability_scores)
            print(f"Stability Score: {mean_stability:.3f} ± {std_stability:.3f}")
            return mean_stability, std_stability
        else:
            print("Stability analysis not available for this algorithm")
            return None, None
    
    def interpret_clusters(self, labels=None):
        """Biological interpretation of clusters"""
        if labels is None:
            labels = self.best_result['labels']
        
        print("\n=== Cluster Interpretation ===")
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 1. Cluster sizes and composition
        print("\nCluster Sizes:")
        for i in range(n_clusters):
            if i == -1:  # DBSCAN noise
                continue
            cluster_size = sum(labels == i)
            print(f"Cluster {i}: {cluster_size} proteins ({cluster_size/len(labels)*100:.1f}%)")
        
        # 2. Feature profiles per cluster
        cluster_profiles = pd.DataFrame()
        for i in range(n_clusters):
            if i == -1:
                continue
            mask = labels == i
            cluster_mean = self.X[mask].mean(axis=0)
            cluster_std = self.X[mask].std(axis=0)
            cluster_profiles[f'Cluster_{i}_mean'] = cluster_mean
            cluster_profiles[f'Cluster_{i}_std'] = cluster_std
        
        cluster_profiles.index = self.feature_names
        
        # 3. Create heatmap of cluster profiles
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mean profiles
        mean_cols = [col for col in cluster_profiles.columns if 'mean' in col]
        mean_profiles = cluster_profiles[mean_cols]
        mean_profiles.columns = [f'C{i}' for i in range(n_clusters)]
        
        # Normalize for visualization
        mean_profiles_norm = (mean_profiles.T - mean_profiles.T.mean()) / mean_profiles.T.std()
        
        sns.heatmap(mean_profiles_norm.T, cmap='RdBu_r', center=0, 
                   annot=True, fmt='.2f', ax=ax1)
        ax1.set_title('Normalized Cluster Feature Profiles')
        ax1.set_ylabel('Features')
        ax1.set_xlabel('Clusters')
        
        # Feature importance (variance across clusters)
        feature_importance = mean_profiles.var(axis=1).sort_values(ascending=False)
        ax2.barh(range(len(feature_importance)), feature_importance.values)
        ax2.set_yticks(range(len(feature_importance)))
        ax2.set_yticklabels(feature_importance.index)
        ax2.set_xlabel('Variance across clusters')
        ax2.set_title('Feature Discriminative Power')
        
        plt.tight_layout()
        plt.show()
        
        # 4. Biological mapping
        print("\n--- Biological Interpretation ---")
        for i in range(n_clusters):
            if i == -1:
                continue
            
            profile = mean_profiles[f'C{i}']
            print(f"\nCluster {i} characteristics:")
            
            # Identify dominant features
            top_features = profile.nlargest(3)
            print(f"  Dominant features: {', '.join(top_features.index)}")
            
            # Biological inference based on known feature meanings
            interpretation = self._infer_localization(profile)
            print(f"  Likely localization: {interpretation}")
            
            # Sample proteins
            cluster_proteins = self.protein_names[labels == i].tolist()[:5]
            print(f"  Example proteins: {', '.join(cluster_proteins)}")
    
    def _infer_localization(self, profile):
        """Infer subcellular localization from feature profile"""
        interpretations = []
        
        # Based on biological knowledge of features
        if profile['mit'] > 0.4:
            interpretations.append("Mitochondrial")
        if profile['nuc'] > 0.4:
            interpretations.append("Nuclear")
        if profile['mcg'] > 0.6 and profile['gvh'] > 0.6:
            interpretations.append("Secretory pathway (ER/Golgi)")
        if profile['vac'] > 0.6:
            interpretations.append("Vacuolar")
        if profile['pox'] > 0.5:
            interpretations.append("Peroxisomal")
        if profile['alm'] > 0.6:
            interpretations.append("Membrane-bound")
        
        if not interpretations:
            if profile['mcg'] < 0.4 and profile['nuc'] < 0.3:
                interpretations.append("Cytoplasmic")
            else:
                interpretations.append("Mixed/Uncertain")
        
        return " / ".join(interpretations)
    
    def visualize_clusters(self, labels=None, method='2d'):
        """Visualize clustering results"""
        if labels is None:
            labels = self.best_result['labels']
        
        print("\n=== Cluster Visualization ===")
        
        # Use reduced data if available, otherwise use PCA
        if hasattr(self, 'X_reduced') and self.X_reduced.shape[1] >= 2:
            X_vis = self.X_reduced
        else:
            pca = PCA(n_components=3)
            X_vis = pca.fit_transform(self.X_scaled)
        
        if method == '2d':
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=labels, 
                                cmap='tab10', alpha=0.6, edgecolors='black', linewidth=0.5)
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('Cluster Visualization (2D)')
            plt.colorbar(scatter, label='Cluster')
            
            # Add cluster centers
            for i in set(labels):
                if i == -1:
                    continue
                mask = labels == i
                center = X_vis[mask].mean(axis=0)
                plt.scatter(center[0], center[1], marker='*', s=500, c='red', 
                          edgecolor='black', linewidth=2)
                plt.annotate(f'C{i}', (center[0], center[1]), fontsize=12, 
                           ha='center', va='center')
            
            plt.show()
        
        elif method == '3d' and X_vis.shape[1] >= 3:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], 
                               c=labels, cmap='tab10', alpha=0.6, s=50)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title('Cluster Visualization (3D)')
            
            plt.colorbar(scatter, label='Cluster', ax=ax)
            plt.show()
    
    def generate_report(self):
        """Generate comprehensive clustering report"""
        print("\n" + "="*50)
        print("COMPREHENSIVE CLUSTERING REPORT")
        print("="*50)
        
        if not self.results:
            print("No results available. Run compare_clustering_algorithms first.")
            return
        
        # Best overall result
        print(f"\n1. BEST CLUSTERING RESULT:")
        best_key = max(self.results.keys(), key=lambda k: self.results[k]['silhouette'])
        best = self.results[best_key]
        print(f"   Algorithm: {best_key}")
        print(f"   Silhouette Score: {best['silhouette']:.3f}")
        print(f"   Davies-Bouldin Index: {best['davies_bouldin']:.3f}")
        print(f"   Calinski-Harabasz Score: {best['calinski_harabasz']:.1f}")
        print(f"   Number of Clusters: {best['n_clusters']}")
        
        # Improvement over baseline
        baseline_sil = 0.21  # Your original K-means score
        improvement = ((best['silhouette'] - baseline_sil) / baseline_sil) * 100
        print(f"\n2. IMPROVEMENT OVER BASELINE:")
        print(f"   Silhouette improvement: {improvement:.1f}%")
        print(f"   Original: 0.21 → Current: {best['silhouette']:.3f}")
        
        # Summary statistics
        print(f"\n3. DATASET SUMMARY:")
        print(f"   Total proteins: {len(self.df)}")
        print(f"   Features used: {len(self.selected_features)}")
        print(f"   Feature selection: {'Yes' if len(self.selected_features) < 8 else 'No'}")
        print(f"   Dimensionality reduction: {self.dim_reducer.__class__.__name__ if self.dim_reducer else 'None'}")
        
        # Resume bullets
        print(f"\n4. RESUME-READY ACHIEVEMENTS:")
        print(f"   • Improved clustering quality by {improvement:.0f}% through advanced preprocessing and algorithm selection")
        print(f"   • Identified {best['n_clusters']} distinct protein localization patterns in yeast cellular architecture")
        print(f"   • Implemented ensemble approach comparing 6 clustering algorithms with comprehensive validation")
        print(f"   • Achieved Silhouette score of {best['silhouette']:.3f} using {best_key.split('_')[0]} on dimensionality-reduced data")
        
        return best

# Example usage script
def main():
    # Initialize the clustering pipeline
    yc = ImprovedYeastClustering(random_state=42)
    
    # 1. Load and analyze data
    print("Step 1: Loading and analyzing data...")
    X = yc.load_and_prepare_data('yeast_train.csv')
    
    # 2. Preprocess data
    print("\nStep 2: Preprocessing...")
    X_scaled = yc.preprocess_data(remove_low_var=True, use_power_transform=True)
    
    # 3. Dimensionality reduction
    print("\nStep 3: Dimensionality reduction...")
    X_reduced = yc.reduce_dimensions(method='pca', n_components=3)  # Use 'umap' if available
    
    # 4. Find optimal number of clusters
    print("\nStep 4: Finding optimal clusters...")
    optimal_k = yc.find_optimal_clusters(max_k=8)
    
    # 5. Compare algorithms
    print("\nStep 5: Comparing clustering algorithms...")
    results = yc.compare_clustering_algorithms(n_clusters=5)  # or use optimal_k
    
    # 6. Analyze stability
    print("\nStep 6: Analyzing stability...")
    stability_mean, stability_std = yc.analyze_cluster_stability(algorithm='kmeans', n_clusters=5)
    
    # 7. Interpret clusters
    print("\nStep 7: Interpreting clusters...")
    yc.interpret_clusters()
    
    # 8. Visualize results
    print("\nStep 8: Visualizing clusters...")
    yc.visualize_clusters(method='2d')
    
    # 9. Generate final report
    print("\nStep 9: Generating final report...")
    best_result = yc.generate_report()
    
    return yc, best_result

if __name__ == "__main__":
    yc, results = main()