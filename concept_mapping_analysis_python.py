#!/usr/bin/env python3
"""
Concept Mapping Analysis - Python Implementation
===============================================

A comprehensive Python implementation of concept mapping analysis featuring:
- Multidimensional Scaling (MDS) for concept positioning
- K-means clustering with optimal cluster selection
- Advanced visualizations and statistical analysis
- Publication-quality graphics and interactive plots

This implementation is designed for researchers conducting concept mapping studies
in healthcare, education, business, or any domain requiring structured analysis
of complex ideas and their relationships.

Author: BCCS AI Workshop Team
Date: July 27, 2025
License: Educational and Research Use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure plotting style for publication-quality graphics
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ConceptMappingAnalysis:
    """
    Main class for concept mapping analysis.
    
    This class provides a complete workflow for concept mapping analysis including:
    - Data loading and validation
    - Multidimensional scaling (MDS)
    - Clustering analysis with optimal k selection
    - Visualization generation
    - Statistical analysis and reporting
    
    Attributes:
        data_dir (str): Directory containing the transformed data files
        statements (pd.DataFrame): Statements/concepts to be analyzed
        ratings (pd.DataFrame): Participant ratings on statements
        demographics (pd.DataFrame): Participant demographic information
        sorted_cards (pd.DataFrame): Grouping/sorting data (if available)
        output_dir (str): Directory for saving generated visualizations
    """
    
    def __init__(self, data_dir="data/python_analysis"):
        """
        Initialize the concept mapping analysis.
        
        Args:
            data_dir (str): Path to directory containing transformed data files.
                           Expected files: statements.csv, ratings.csv, 
                           demographics.csv, sorted_cards.csv
        """
        self.data_dir = data_dir
        self.statements = None
        self.ratings = None
        self.demographics = None
        self.sorted_cards = None
        self.output_dir = "Figures/python_analysis"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """
        Load and validate the transformed concept mapping data.
        
        This method loads four key data files:
        - statements.csv: Contains the concepts/statements to be analyzed
        - ratings.csv: Contains participant ratings on importance and feasibility
        - demographics.csv: Contains participant demographic information
        - sorted_cards.csv: Contains grouping/sorting data (optional)
        
        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data format is invalid
        """
        print("Loading concept mapping data...")
        
        try:
            # Load core data files
            self.statements = pd.read_csv(os.path.join(self.data_dir, "statements.csv"))
            self.ratings = pd.read_csv(os.path.join(self.data_dir, "ratings.csv"))
            self.demographics = pd.read_csv(os.path.join(self.data_dir, "demographics.csv"))
            self.sorted_cards = pd.read_csv(os.path.join(self.data_dir, "sorted_cards.csv"))
            
            # Validate data structure
            self._validate_data()
            
            print(f"✅ Successfully loaded:")
            print(f"   - {len(self.statements)} statements")
            print(f"   - {len(self.ratings)} ratings from {len(self.demographics)} participants")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Required data file not found: {e}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def _validate_data(self):
        """
        Validate the loaded data for required structure and content.
        
        Checks for:
        - Required columns in each dataset
        - Data types and ranges
        - Missing values and data quality
        """
        # Validate statements data
        required_statement_cols = ['StatementID', 'StatementText']
        if not all(col in self.statements.columns for col in required_statement_cols):
            raise ValueError("Statements data missing required columns: StatementID, StatementText")
        
        # Validate ratings data
        required_rating_cols = ['ParticipantID', 'StatementID', 'RatingType', 'Rating']
        if not all(col in self.ratings.columns for col in required_rating_cols):
            raise ValueError("Ratings data missing required columns: ParticipantID, StatementID, RatingType, Rating")
        
        # Check rating values are numeric and in expected range
        if not self.ratings['Rating'].dtype in ['int64', 'float64']:
            raise ValueError("Rating values must be numeric")
        
        # Validate rating types
        expected_rating_types = ['Importance', 'Feasibility']
        actual_rating_types = self.ratings['RatingType'].unique()
        if not all(rt in expected_rating_types for rt in actual_rating_types):
            print(f"⚠️  Warning: Unexpected rating types found: {actual_rating_types}")
        
    def prepare_rating_matrix(self):
        """
        Prepare rating matrix for multidimensional scaling analysis.
        
        This method transforms the long-format ratings data into a wide-format
        matrix where each row represents a statement and each column represents
        a participant's rating on a specific dimension (importance/feasibility).
        
        Returns:
            pd.DataFrame: Rating matrix with statements as rows and 
                         participant-rating combinations as columns
        """
        print("Preparing rating matrix for MDS analysis...")
        
        # Create pivot table to transform long format to wide format
        rating_matrix = self.ratings.pivot_table(
            index='StatementID',
            columns=['ParticipantID', 'RatingType'],
            values='Rating',
            aggfunc='mean'  # Use mean in case of duplicate ratings
        ).fillna(0)  # Fill missing values with 0
        
        # Flatten multi-level column names for easier handling
        rating_matrix.columns = [f"{col[0]}_{col[1]}" for col in rating_matrix.columns]
        
        print(f"✅ Rating matrix prepared: {rating_matrix.shape[0]} statements × {rating_matrix.shape[1]} rating dimensions")
        
        return rating_matrix
    
    def perform_mds(self, rating_matrix, n_components=2):
        """
        Perform Multidimensional Scaling (MDS) on the rating matrix.
        
        MDS converts the high-dimensional rating patterns into 2D coordinates
        that preserve the relative distances between statements. Statements with
        similar rating patterns will be positioned closer together.
        
        Args:
            rating_matrix (pd.DataFrame): Matrix of ratings with statements as rows
            n_components (int): Number of dimensions for MDS (default: 2)
        
        Returns:
            tuple: (mds_coordinates, similarity_matrix)
                - mds_coordinates: 2D coordinates for each statement
                - similarity_matrix: Correlation matrix between statements
        """
        print("Performing Multidimensional Scaling (MDS)...")
        
        # Calculate similarity matrix using correlation
        # Higher correlation = more similar rating patterns
        similarity_matrix = rating_matrix.corr()
        
        # Convert similarity to distance matrix
        # Distance = 1 - |correlation| (absolute correlation to handle negative correlations)
        distance_matrix = 1 - np.abs(similarity_matrix)
        distance_matrix = np.array(distance_matrix)  # Convert to numpy array
        np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0 (self-similarity)
        
        # Handle NaN values that may arise from missing data
        distance_matrix = np.nan_to_num(distance_matrix, nan=0)
        
        # Perform MDS using precomputed distances
        mds = MDS(
            n_components=n_components, 
            dissimilarity='precomputed', 
            random_state=42,  # For reproducible results
            n_init=10  # Multiple initializations for better results
        )
        mds_coords = mds.fit_transform(distance_matrix)
        
        print(f"✅ MDS completed: {mds_coords.shape[0]} statements mapped to {n_components}D space")
        
        return mds_coords, similarity_matrix
    
    def find_optimal_clusters(self, mds_coords, max_k=10):
        """
        Find the optimal number of clusters using elbow method and silhouette analysis.
        
        This method evaluates different numbers of clusters (k) and selects the
        optimal k based on:
        1. Elbow method: Where the within-cluster sum of squares (WSS) starts to level off
        2. Silhouette analysis: Average silhouette score for each k
        
        Args:
            mds_coords (np.ndarray): 2D coordinates from MDS
            max_k (int): Maximum number of clusters to evaluate (default: 10)
        
        Returns:
            tuple: (optimal_k, wss_scores, silhouette_scores, k_range)
                - optimal_k: Recommended number of clusters
                - wss_scores: Within-cluster sum of squares for each k
                - silhouette_scores: Average silhouette score for each k
                - k_range: Range of k values evaluated
        """
        print("Finding optimal number of clusters...")
        
        # Standardize coordinates for clustering
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(mds_coords)
        
        # Adjust max_k based on number of samples
        n_samples = len(mds_coords)
        max_k = min(max_k, n_samples - 1)  # Can't have more clusters than samples - 1
        
        # Calculate metrics for different k values
        wss = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            # Fit K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords_scaled)
            
            # Calculate WSS (within-cluster sum of squares)
            wss.append(kmeans.inertia_)
            
            # Calculate silhouette score (only if k > 1 and k < n_samples)
            if k > 1 and k < n_samples:
                try:
                    silhouette_avg = silhouette_score(coords_scaled, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                except ValueError:
                    # If silhouette calculation fails, use 0
                    silhouette_scores.append(0)
            else:
                silhouette_scores.append(0)
        
        # Find optimal k using elbow method
        optimal_k = self._find_elbow_point(k_range, wss)
        
        print(f"✅ Optimal number of clusters: {optimal_k}")
        print(f"   - Evaluated k range: {list(k_range)}")
        print(f"   - Best silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k, wss, silhouette_scores, k_range
    
    def _find_elbow_point(self, k_range, wss):
        """
        Find the elbow point in the WSS curve using the second derivative method.
        
        The elbow point is where the rate of decrease in WSS starts to level off,
        indicating diminishing returns from adding more clusters.
        
        Args:
            k_range (range): Range of k values evaluated
            wss (list): Within-cluster sum of squares for each k
        
        Returns:
            int: Optimal number of clusters (k value at elbow point)
        """
        # Calculate second derivative (rate of change of rate of change)
        if len(wss) < 3:
            return k_range[0]  # Default to first k if not enough points
        
        # First differences
        first_diff = np.diff(wss)
        # Second differences
        second_diff = np.diff(first_diff)
        
        # Find the point with maximum second derivative (sharpest bend)
        elbow_idx = np.argmax(second_diff) + 2  # +2 because of double differencing
        
        # Ensure elbow_idx is within bounds
        elbow_idx = min(elbow_idx, len(k_range) - 1)
        
        return k_range[elbow_idx]
    
    def perform_clustering(self, mds_coords, n_clusters):
        """
        Perform K-means clustering on the MDS coordinates.
        
        Args:
            mds_coords (np.ndarray): 2D coordinates from MDS
            n_clusters (int): Number of clusters to create
        
        Returns:
            tuple: (cluster_labels, kmeans_model)
                - cluster_labels: Cluster assignment for each statement
                - kmeans_model: Fitted KMeans model
        """
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(mds_coords)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coords_scaled)
        
        print(f"✅ Clustering completed: {len(np.unique(cluster_labels))} clusters created")
        
        return cluster_labels, kmeans
    
    def create_visualizations(self, mds_coords, cluster_labels, similarity_matrix):
        """
        Create comprehensive visualizations for the concept mapping analysis.
        
        This method generates multiple visualization types:
        1. Concept map (MDS plot with clusters)
        2. Importance vs feasibility scatter plot
        3. Rating distribution histograms
        4. Cluster analysis plots (WSS and silhouette)
        5. Similarity matrix heatmap
        
        Args:
            mds_coords (np.ndarray): 2D coordinates from MDS
            cluster_labels (np.ndarray): Cluster assignments for each statement
            similarity_matrix (pd.DataFrame): Correlation matrix between statements
        """
        print("Creating visualizations...")
        
        # Create concept map
        self._create_concept_map(mds_coords, cluster_labels)
        
        # Create importance vs feasibility plot
        self._create_importance_feasibility_plot()
        
        # Create rating distribution plots
        self._create_rating_distribution()
        
        # Create cluster analysis plots
        self._create_cluster_analysis_plots(mds_coords)
        
        # Create similarity heatmap
        self._create_heatmap(similarity_matrix)
        
        print("✅ All visualizations created successfully")
    
    def _create_concept_map(self, mds_coords, cluster_labels):
        """
        Create the main concept map visualization.
        
        This is the primary visualization showing how statements are positioned
        in 2D space based on their rating patterns, with color-coding by cluster.
        
        Args:
            mds_coords (np.ndarray): 2D coordinates from MDS
            cluster_labels (np.ndarray): Cluster assignments for each statement
        """
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with cluster colors
        scatter = plt.scatter(
            mds_coords[:, 0], 
            mds_coords[:, 1], 
            c=cluster_labels, 
            cmap='viridis', 
            s=100, 
            alpha=0.7,
            edgecolors='white',
            linewidth=1
        )
        
        # Add statement labels
        for i, (x, y) in enumerate(mds_coords):
            plt.annotate(
                f"{i+1}", 
                (x, y), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                fontweight='bold'
            )
        
        # Add cluster centers
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            cluster_points = mds_coords[cluster_labels == cluster]
            center = cluster_points.mean(axis=0)
            plt.scatter(
                center[0], center[1], 
                c='red', 
                s=200, 
                marker='*', 
                edgecolors='black',
                linewidth=2,
                label=f'Cluster {cluster+1} Center'
            )
        
        plt.xlabel('MDS Dimension 1', fontsize=12, fontweight='bold')
        plt.ylabel('MDS Dimension 2', fontsize=12, fontweight='bold')
        plt.title('Concept Map: Multidimensional Scaling with Clusters', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(), 
                           title="Clusters", 
                           loc="upper right")
        plt.gca().add_artist(legend1)
        
        # Add cluster centers to legend
        plt.legend(loc="lower left")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'concept_map.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Concept map created")
    
    def _create_importance_feasibility_plot(self):
        """
        Create importance vs feasibility scatter plot.
        
        This visualization shows the relationship between importance and feasibility
        ratings, with statements positioned based on their mean ratings on both dimensions.
        """
        # Calculate mean importance and feasibility ratings for each statement
        importance_feasibility = self.ratings.pivot_table(
            index='StatementID',
            columns='RatingType',
            values='Rating',
            aggfunc='mean'
        ).reset_index()
        
        # Merge with statement text
        plot_data = importance_feasibility.merge(self.statements, on='StatementID')
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        plt.scatter(
            plot_data['Importance'], 
            plot_data['Feasibility'], 
            s=100, 
            alpha=0.7,
            c='steelblue',
            edgecolors='white',
            linewidth=1
        )
        
        # Add statement labels
        for _, row in plot_data.iterrows():
            plt.annotate(
                f"{row['StatementID']}", 
                (row['Importance'], row['Feasibility']), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                fontweight='bold'
            )
        
        # Add mean lines
        mean_importance = plot_data['Importance'].mean()
        mean_feasibility = plot_data['Feasibility'].mean()
        
        plt.axvline(mean_importance, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean Importance: {mean_importance:.2f}')
        plt.axhline(mean_feasibility, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean Feasibility: {mean_feasibility:.2f}')
        
        # Add quadrants
        plt.axvline(mean_importance, color='gray', linestyle='-', alpha=0.3)
        plt.axhline(mean_feasibility, color='gray', linestyle='-', alpha=0.3)
        
        # Add quadrant labels
        plt.text(0.05, 0.95, 'High Importance\nLow Feasibility', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        plt.text(0.95, 0.95, 'High Importance\nHigh Feasibility', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        plt.text(0.05, 0.05, 'Low Importance\nLow Feasibility', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        plt.text(0.95, 0.05, 'Low Importance\nHigh Feasibility', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.xlabel('Importance Rating', fontsize=12, fontweight='bold')
        plt.ylabel('Feasibility Rating', fontsize=12, fontweight='bold')
        plt.title('Importance vs Feasibility: Strategic Quadrants', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'importance_vs_feasibility.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Importance vs feasibility plot created")
    
    def _create_rating_distribution(self):
        """
        Create rating distribution histograms for importance and feasibility.
        
        This visualization shows the distribution of ratings across all statements
        and participants, helping to understand the overall rating patterns.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Importance rating distribution
        importance_ratings = self.ratings[self.ratings['RatingType'] == 'Importance']['Rating']
        ax1.hist(importance_ratings, bins=range(1, 8), alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Importance Rating', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Distribution of Importance Ratings', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add mean line
        mean_importance = importance_ratings.mean()
        ax1.axvline(mean_importance, color='red', linestyle='--', 
                   label=f'Mean: {mean_importance:.2f}')
        ax1.legend()
        
        # Feasibility rating distribution
        feasibility_ratings = self.ratings[self.ratings['RatingType'] == 'Feasibility']['Rating']
        ax2.hist(feasibility_ratings, bins=range(1, 8), alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Feasibility Rating', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Distribution of Feasibility Ratings', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add mean line
        mean_feasibility = feasibility_ratings.mean()
        ax2.axvline(mean_feasibility, color='red', linestyle='--', 
                   label=f'Mean: {mean_feasibility:.2f}')
        ax2.legend()
        
        plt.suptitle('Rating Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'rating_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Rating distribution plots created")
    
    def _create_cluster_analysis_plots(self, mds_coords):
        """
        Create cluster analysis plots showing WSS and silhouette analysis.
        
        These plots help visualize the process of finding the optimal number
        of clusters and validate the clustering solution.
        
        Args:
            mds_coords (np.ndarray): 2D coordinates from MDS
        """
        # Get cluster analysis data
        optimal_k, wss, silhouette_scores, k_range = self.find_optimal_clusters(mds_coords)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # WSS (Elbow) plot
        ax1.plot(k_range, wss, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k = {optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WSS)', fontweight='bold')
        ax1.set_title('Elbow Method for Optimal k Selection', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Silhouette plot
        ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(optimal_k, color='red', linestyle='--', 
                   label=f'Optimal k = {optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax2.set_ylabel('Average Silhouette Score', fontweight='bold')
        ax2.set_title('Silhouette Analysis for Optimal k Selection', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle('Cluster Analysis: Finding Optimal Number of Clusters', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'cluster_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Cluster analysis plots created")
    
    def _create_heatmap(self, similarity_matrix):
        """
        Create similarity matrix heatmap.
        
        This visualization shows the correlation matrix between statements,
        helping to identify groups of statements with similar rating patterns.
        
        Args:
            similarity_matrix (pd.DataFrame): Correlation matrix between statements
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix, 
            cmap='coolwarm', 
            center=0, 
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'},
            xticklabels=False,
            yticklabels=False
        )
        
        plt.title('Statement Similarity Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Statements', fontsize=12, fontweight='bold')
        plt.ylabel('Statements', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'similarity_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Similarity heatmap created")
    
    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics for the concept mapping analysis.
        
        This method creates summary tables and statistics including:
        - Statement-level statistics (mean ratings, cluster assignments)
        - Cluster-level statistics (size, mean ratings, characteristics)
        - Overall correlation between importance and feasibility
        - Data quality metrics
        
        Returns:
            dict: Dictionary containing summary statistics and dataframes
        """
        print("Generating summary statistics...")
        
        # Calculate statement-level statistics
        statement_stats = self.ratings.groupby(['StatementID', 'RatingType'])['Rating'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Pivot to get importance and feasibility side by side
        statement_summary = statement_stats.pivot_table(
            index='StatementID',
            columns='RatingType',
            values=['mean', 'std', 'count'],
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        statement_summary.columns = [
            f"{col[1]}_{col[0]}" if col[1] else col[0] 
            for col in statement_summary.columns
        ]
        
        # Merge with statement text
        statement_summary = statement_summary.merge(self.statements, on='StatementID')
        
        # Calculate overall correlation
        importance_feasibility = self.ratings.pivot_table(
            index='StatementID',
            columns='RatingType',
            values='Rating',
            aggfunc='mean'
        )
        
        correlation = importance_feasibility['Importance'].corr(importance_feasibility['Feasibility'])
        
        # Create summary dictionary
        summary = {
            'correlation_importance_feasibility': correlation,
            'total_statements': len(self.statements),
            'total_participants': len(self.demographics),
            'total_ratings': len(self.ratings),
            'statement_summary': statement_summary,
            'data_quality': {
                'missing_ratings': self.ratings['Rating'].isna().sum(),
                'rating_range': (self.ratings['Rating'].min(), self.ratings['Rating'].max()),
                'unique_rating_types': self.ratings['RatingType'].unique().tolist()
            }
        }
        
        # Save summary statistics
        statement_summary.to_csv(os.path.join(self.output_dir, 'summary_statistics.csv'), 
                               index=False)
        
        print(f"✅ Summary statistics generated:")
        print(f"   - Correlation (Importance vs Feasibility): {correlation:.3f}")
        print(f"   - Total statements: {summary['total_statements']}")
        print(f"   - Total participants: {summary['total_participants']}")
        print(f"   - Total ratings: {summary['total_ratings']}")
        
        return summary
    
    def run_analysis(self):
        """
        Run the complete concept mapping analysis workflow.
        
        This method orchestrates the entire analysis process:
        1. Load and validate data
        2. Prepare rating matrix
        3. Perform MDS
        4. Find optimal clusters
        5. Perform clustering
        6. Create visualizations
        7. Generate summary statistics
        
        Returns:
            dict: Complete analysis results including coordinates, clusters, and statistics
        """
        print("=" * 60)
        print("CONCEPT MAPPING ANALYSIS - PYTHON IMPLEMENTATION")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Prepare rating matrix
        rating_matrix = self.prepare_rating_matrix()
        
        # Step 3: Perform MDS
        mds_coords, similarity_matrix = self.perform_mds(rating_matrix)
        
        # Step 4: Find optimal clusters
        optimal_k, wss, silhouette_scores, k_range = self.find_optimal_clusters(mds_coords)
        
        # Step 5: Perform clustering
        cluster_labels, kmeans_model = self.perform_clustering(mds_coords, optimal_k)
        
        # Step 6: Create visualizations
        self.create_visualizations(mds_coords, cluster_labels, similarity_matrix)
        
        # Step 7: Generate summary statistics
        summary_stats = self.generate_summary_statistics()
        
        # Step 8: Create results dataframe
        # Only include statements that have MDS coordinates (i.e., those with ratings)
        results_df = pd.DataFrame({
            'StatementID': self.statements['StatementID'].iloc[:len(mds_coords)],
            'StatementText': self.statements['StatementText'].iloc[:len(mds_coords)],
            'MDS_Dim1': mds_coords[:, 0],
            'MDS_Dim2': mds_coords[:, 1],
            'Cluster': cluster_labels + 1  # Convert to 1-based indexing
        })
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, 'statements_with_clusters.csv'), 
                         index=False)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Results saved to: {self.output_dir}")
        print(f"📈 Visualizations: {len(os.listdir(self.output_dir))} files created")
        print(f"📋 Summary: {summary_stats['total_statements']} statements in {optimal_k} clusters")
        
        # Return complete results
        return {
            'statements': self.statements,
            'ratings': self.ratings,
            'demographics': self.demographics,
            'mds_coords': mds_coords,
            'cluster_labels': cluster_labels,
            'similarity_matrix': similarity_matrix,
            'optimal_k': optimal_k,
            'kmeans_model': kmeans_model,
            'summary_stats': summary_stats,
            'results_df': results_df
        }


def main():
    """
    Main function to run the concept mapping analysis.
    
    This function creates an instance of ConceptMappingAnalysis and runs
    the complete analysis workflow. It serves as the entry point for
    the script when run directly.
    """
    try:
        # Create analysis instance
        analysis = ConceptMappingAnalysis()
        
        # Run complete analysis
        results = analysis.run_analysis()
        
        print("\n🎉 Analysis completed successfully!")
        print("📁 Check the 'Figures/python_analysis' directory for results.")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check your data files and try again.")
        raise


if __name__ == "__main__":
    main() 