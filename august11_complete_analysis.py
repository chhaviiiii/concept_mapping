import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
import warnings
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
warnings.filterwarnings('ignore')

# Set the same color scheme as July 27 analysis
plt.style.use('default')
# Updated color scheme to match the preferred images
colors = ['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
sns.set_palette(colors)

class August11CompleteAnalysis:
    def __init__(self):
        """Initialize the complete analysis for August 11 data."""
        self.data_dir = "data/rcmap_august11_2025"
        self.output_dir = "Figures/august11_2025_analysis"
        self.custom_dir = "Figures/custom_graphs_august11"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.custom_dir, exist_ok=True)
        
        # Load data
        self.statements_df = pd.read_csv(f'{self.data_dir}/Statements.csv')
        self.ratings_df = pd.read_csv(f'{self.data_dir}/Ratings.csv')
        self.sorted_cards_df = pd.read_csv(f'{self.data_dir}/SortedCards.csv')
        self.demographics_df = pd.read_csv(f'{self.data_dir}/Demographics.csv')
        
        # Create importance-feasibility matrix
        self.create_matrix()
        
        print(f"Loaded August 11 data for complete analysis:")
        print(f"  - {len(self.statements_df)} statements")
        print(f"  - {len(self.ratings_df)} ratings")
        print(f"  - {len(self.sorted_cards_df)} groupings")
        print(f"  - {len(self.demographics_df)} participants")
    
    def create_matrix(self):
        """Create importance vs feasibility matrix."""
        # Pivot data to get importance and feasibility for each statement
        importance_data = self.ratings_df[self.ratings_df['RatingType'] == 'Importance'].groupby('StatementID')['Rating'].agg(['mean', 'std', 'count']).reset_index()
        feasibility_data = self.ratings_df[self.ratings_df['RatingType'] == 'Feasibility'].groupby('StatementID')['Rating'].agg(['mean', 'std', 'count']).reset_index()
        
        # Merge with statements
        importance_data = importance_data.merge(self.statements_df, on='StatementID')
        feasibility_data = feasibility_data.merge(self.statements_df, on='StatementID')
        
        # Create combined matrix
        self.matrix_data = importance_data[['StatementID', 'StatementText', 'mean', 'std', 'count']].copy()
        self.matrix_data.columns = ['StatementID', 'StatementText', 'Importance_Mean', 'Importance_Std', 'Importance_Count']
        self.matrix_data = self.matrix_data.merge(feasibility_data[['StatementID', 'mean', 'std', 'count']], on='StatementID')
        self.matrix_data.columns = ['StatementID', 'StatementText', 'Importance_Mean', 'Importance_Std', 'Importance_Count', 'Feasibility_Mean', 'Feasibility_Std', 'Feasibility_Count']
        
        # Calculate gap (importance - feasibility)
        self.matrix_data['Gap'] = self.matrix_data['Importance_Mean'] - self.matrix_data['Feasibility_Mean']
    
    def create_similarity_heatmap(self):
        """Create similarity heatmap between statements."""
        # Create similarity matrix based on importance and feasibility ratings
        similarity_data = self.matrix_data[['Importance_Mean', 'Feasibility_Mean']].values
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(similarity_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create mask for upper triangle to show only lower triangle
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
        
        # Create heatmap with better color scheme
        sns.heatmap(similarity_matrix, 
                   mask=mask,
                   cmap='YlOrRd', 
                   center=0.5,
                   square=True,
                   cbar_kws={'label': 'Similarity Score'},
                   ax=ax)
        
        # Add statement IDs as labels
        statement_ids = [f"Stmt {row['StatementID']}" for _, row in self.matrix_data.iterrows()]
        ax.set_xticks(np.arange(len(statement_ids)) + 0.5)
        ax.set_yticks(np.arange(len(statement_ids)) + 0.5)
        ax.set_xticklabels(statement_ids, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(statement_ids, fontsize=8)
        
        ax.set_title('Statement Similarity Heatmap - August 11 Dataset\n(Based on Importance and Feasibility Ratings)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.custom_dir}/similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created similarity heatmap")
    
    def create_optimal_clusters_analysis(self):
        """Create comprehensive cluster analysis with optimal number determination."""
        # Prepare data for clustering
        X = self.matrix_data[['Importance_Mean', 'Feasibility_Mean']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test different numbers of clusters
        K_range = range(2, 11)
        wss = []
        silhouette_scores = []
        gap_stats = []
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            
            # Calculate gap statistic
            if k > 1:
                # Generate reference data
                reference_data = np.random.uniform(X_scaled.min(), X_scaled.max(), X_scaled.shape)
                reference_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                reference_kmeans.fit(reference_data)
                gap_stat = np.log(reference_kmeans.inertia_) - np.log(kmeans.inertia_)
                gap_stats.append(gap_stat)
            else:
                gap_stats.append(0)
        
        # Create comprehensive cluster analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Elbow method
        ax1.plot(K_range, wss, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Within-Cluster Sum of Squares (WSS)', fontsize=12)
        ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette analysis
        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Gap statistic
        ax3.plot(K_range[1:], gap_stats[1:], 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax3.set_ylabel('Gap Statistic', fontsize=12)
        ax3.set_title('Gap Statistic Analysis', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Combined analysis
        ax4.plot(K_range, wss, 'b-', label='WSS', linewidth=2)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(K_range, silhouette_scores, 'r-', label='Silhouette', linewidth=2)
        ax4.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax4.set_ylabel('WSS', fontsize=12, color='blue')
        ax4_twin.set_ylabel('Silhouette Score', fontsize=12, color='red')
        ax4.set_title('Combined Cluster Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/optimal_clusters_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Determine optimal k (you can adjust this based on the plots)
        optimal_k = 4  # Based on typical concept mapping results
        
        # Perform final clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.matrix_data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create cluster summary (individual statement assignments) - matching July 27 format
        cluster_summary = self.matrix_data[['StatementID', 'StatementText', 'Cluster']].copy()
        cluster_summary = cluster_summary.sort_values(['Cluster', 'StatementID'])
        cluster_summary.to_csv(f'{self.output_dir}/cluster_summary.csv', index=False)
        
        # Create cluster ratings (aggregated cluster statistics) - matching July 27 format
        cluster_stats = self.matrix_data.groupby('Cluster').agg({
            'Importance_Mean': 'mean',
            'Feasibility_Mean': 'mean',
            'StatementID': 'count'
        }).round(6)
        cluster_stats.columns = ['mean_importance', 'mean_feasibility', 'n_statements']
        cluster_stats = cluster_stats.reset_index()
        cluster_stats.to_csv(f'{self.output_dir}/cluster_ratings.csv', index=False)
        
        print(f"Created optimal clusters analysis with {optimal_k} clusters")
        return optimal_k
    
    def create_improved_quadrant_analysis(self):
        """Create improved quadrant analysis with better color scheme."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Calculate medians for quadrant boundaries
        importance_median = self.matrix_data['Importance_Mean'].median()
        feasibility_median = self.matrix_data['Feasibility_Mean'].median()
        
        # Define quadrants with better colors
        quadrants = {
            'Q1': {'name': 'High Importance, High Feasibility', 'color': '#1f77b4', 'alpha': 0.7},  # Blue
            'Q2': {'name': 'High Importance, Low Feasibility', 'color': '#9467bd', 'alpha': 0.7},  # Purple
            'Q3': {'name': 'Low Importance, Low Feasibility', 'color': '#7f7f7f', 'alpha': 0.7},  # Gray
            'Q4': {'name': 'Low Importance, High Feasibility', 'color': '#ff7f0e', 'alpha': 0.7}  # Orange
        }
        
        # Create scatter plot with different colors for each quadrant
        for idx, row in self.matrix_data.iterrows():
            if row['Importance_Mean'] >= importance_median and row['Feasibility_Mean'] >= feasibility_median:
                quadrant = 'Q1'
            elif row['Importance_Mean'] >= importance_median and row['Feasibility_Mean'] < feasibility_median:
                quadrant = 'Q2'
            elif row['Importance_Mean'] < importance_median and row['Feasibility_Mean'] < feasibility_median:
                quadrant = 'Q3'
            else:
                quadrant = 'Q4'
            
            ax.scatter(row['Feasibility_Mean'], row['Importance_Mean'], 
                      s=row['Importance_Mean'] * 50, 
                      c=quadrants[quadrant]['color'], 
                      alpha=quadrants[quadrant]['alpha'], 
                      edgecolors='black', 
                      linewidth=1)
            
            # Add statement ID labels for important points
            if row['Importance_Mean'] > importance_median + 0.5 or row['Feasibility_Mean'] > feasibility_median + 0.5:
                ax.annotate(f"{row['StatementID']}", 
                           (row['Feasibility_Mean'], row['Importance_Mean']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Add quadrant lines
        ax.axhline(y=importance_median, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=feasibility_median, color='black', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add quadrant labels with better styling
        ax.text(0.95, 0.95, 'Q1: High Importance\nHigh Feasibility\n(Immediate Actions)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.8, edgecolor='black'))
        ax.text(0.05, 0.95, 'Q2: High Importance\nLow Feasibility\n(Strategic Planning)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#9467bd', alpha=0.8, edgecolor='black'))
        ax.text(0.05, 0.05, 'Q3: Low Importance\nLow Feasibility\n(Monitor)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='#7f7f7f', alpha=0.8, edgecolor='black'))
        ax.text(0.95, 0.05, 'Q4: Low Importance\nHigh Feasibility\n(Quick Wins)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.8, edgecolor='black'))
        
        # Customize plot
        ax.set_xlabel('Feasibility Rating (Mean)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Importance Rating (Mean)', fontsize=14, fontweight='bold')
        ax.set_title('Quadrant Analysis - August 11 Dataset\n(Size = Importance, Color = Quadrant)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        q1_count = len(self.matrix_data[(self.matrix_data['Importance_Mean'] >= importance_median) & 
                                       (self.matrix_data['Feasibility_Mean'] >= feasibility_median)])
        q2_count = len(self.matrix_data[(self.matrix_data['Importance_Mean'] >= importance_median) & 
                                       (self.matrix_data['Feasibility_Mean'] < feasibility_median)])
        q3_count = len(self.matrix_data[(self.matrix_data['Importance_Mean'] < importance_median) & 
                                       (self.matrix_data['Feasibility_Mean'] < feasibility_median)])
        q4_count = len(self.matrix_data[(self.matrix_data['Importance_Mean'] < importance_median) & 
                                       (self.matrix_data['Feasibility_Mean'] >= feasibility_median)])
        
        stats_text = f'Q1: {q1_count} | Q2: {q2_count} | Q3: {q3_count} | Q4: {q4_count}'
        ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, fontsize=10, 
                horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig(f'{self.custom_dir}/quadrant_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created improved quadrant analysis")
    
    def create_improved_bubble_chart(self):
        """Create improved bubble chart with better color scheme."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create bubble chart with better color scheme
        scatter = ax.scatter(self.matrix_data['Feasibility_Mean'], 
                           self.matrix_data['Importance_Mean'],
                           s=self.matrix_data['Importance_Mean'] * 100,  # Size based on importance
                           c=self.matrix_data['Gap'],  # Color based on gap
                           cmap='RdYlBu',
                           alpha=0.8,
                           edgecolors='black',
                           linewidth=1)
        
        # Add labels for important statements
        for idx, row in self.matrix_data.iterrows():
            if row['Importance_Mean'] > 3.5 or abs(row['Gap']) > 1.0:
                ax.annotate(f"{row['StatementID']}", 
                           (row['Feasibility_Mean'], row['Importance_Mean']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Add diagonal line
        min_val = min(self.matrix_data['Importance_Mean'].min(), self.matrix_data['Feasibility_Mean'].min())
        max_val = max(self.matrix_data['Importance_Mean'].max(), self.matrix_data['Feasibility_Mean'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
        
        # Customize plot
        ax.set_xlabel('Feasibility Rating (Mean)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Importance Rating (Mean)', fontsize=14, fontweight='bold')
        ax.set_title('Bubble Chart - August 11 Dataset\n(Size = Importance, Color = Gap)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Gap (Importance - Feasibility)', fontsize=12)
        
        # Add legend for bubble sizes
        legend_elements = [
            plt.scatter([], [], s=100, c='gray', alpha=0.7, label='Low Importance'),
            plt.scatter([], [], s=300, c='gray', alpha=0.7, label='Medium Importance'),
            plt.scatter([], [], s=500, c='gray', alpha=0.7, label='High Importance')
        ]
        ax.legend(handles=legend_elements, loc='upper left', title='Bubble Size')
        
        plt.tight_layout()
        plt.savefig(f'{self.custom_dir}/bubble_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created improved bubble chart")
    
    def create_improved_heatmap(self):
        """Create improved heatmap with better color scheme."""
        # Create pivot table for heatmap
        importance_pivot = self.ratings_df[self.ratings_df['RatingType'] == 'Importance'].pivot_table(
            index='StatementID', columns='ParticipantID', values='Rating', aggfunc='mean'
        )
        
        feasibility_pivot = self.ratings_df[self.ratings_df['RatingType'] == 'Feasibility'].pivot_table(
            index='StatementID', columns='ParticipantID', values='Rating', aggfunc='mean'
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Importance heatmap with better color scheme
        sns.heatmap(importance_pivot, ax=ax1, cmap='YlOrRd', cbar_kws={'label': 'Importance Rating'})
        ax1.set_title('Importance Ratings Heatmap - August 11', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Participant ID')
        ax1.set_ylabel('Statement ID')
        
        # Feasibility heatmap with better color scheme
        sns.heatmap(feasibility_pivot, ax=ax2, cmap='Blues', cbar_kws={'label': 'Feasibility Rating'})
        ax2.set_title('Feasibility Ratings Heatmap - August 11', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Participant ID')
        ax2.set_ylabel('Statement ID')
        
        plt.tight_layout()
        plt.savefig(f'{self.custom_dir}/heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created improved heatmap")
    
    def create_improved_radar_chart(self):
        """Create improved radar chart with better color scheme."""
        # Get top 5 most important statements
        top_statements = self.matrix_data.nlargest(5, 'Importance_Mean')
        
        # Prepare data for radar chart
        categories = [f"Stmt {row['StatementID']}" for _, row in top_statements.iterrows()]
        importance_values = top_statements['Importance_Mean'].values
        feasibility_values = top_statements['Feasibility_Mean'].values
        
        # Number of variables
        N = len(categories)
        
        # Create angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Plot importance values - ensure arrays have same length
        importance_values_complete = np.concatenate([importance_values, [importance_values[0]]])
        ax.plot(angles, importance_values_complete, 'o-', linewidth=3, label='Importance', color='#9467bd', markersize=8)
        ax.fill(angles, importance_values_complete, alpha=0.25, color='#9467bd')
        
        # Plot feasibility values - ensure arrays have same length
        feasibility_values_complete = np.concatenate([feasibility_values, [feasibility_values[0]]])
        ax.plot(angles, feasibility_values_complete, 'o-', linewidth=3, label='Feasibility', color='#1f77b4', markersize=8)
        ax.fill(angles, feasibility_values_complete, alpha=0.25, color='#1f77b4')
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 5)
        ax.set_title('Top 5 Most Important Statements - August 11 Dataset', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.custom_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created improved radar chart")
    
    def create_improved_cluster_comparison(self):
        """Create improved cluster comparison with better color scheme."""
        # Perform clustering
        X = self.matrix_data[['Importance_Mean', 'Feasibility_Mean']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.matrix_data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create cluster comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Scatter plot with clusters using better colors
        cluster_colors = ['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c']  # Blue, Purple, Orange, Green
        for i in range(4):
            cluster_data = self.matrix_data[self.matrix_data['Cluster'] == i]
            ax1.scatter(cluster_data['Feasibility_Mean'], cluster_data['Importance_Mean'],
                       c=cluster_colors[i], label=f'Cluster {i+1}', alpha=0.7, s=100)
        
        ax1.set_xlabel('Feasibility Rating (Mean)', fontsize=12)
        ax1.set_ylabel('Importance Rating (Mean)', fontsize=12)
        ax1.set_title('Clustering Results - August 11 Dataset', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cluster statistics
        cluster_stats = self.matrix_data.groupby('Cluster').agg({
            'Importance_Mean': ['mean', 'std'],
            'Feasibility_Mean': ['mean', 'std'],
            'Gap': ['mean', 'std'],
            'StatementID': 'count'
        }).round(3)
        
        # Create bar chart of cluster characteristics
        cluster_importance = cluster_stats[('Importance_Mean', 'mean')].values
        cluster_feasibility = cluster_stats[('Feasibility_Mean', 'mean')].values
        cluster_counts = cluster_stats[('StatementID', 'count')].values
        
        x = np.arange(4)
        width = 0.35
        
        ax2.bar(x - width/2, cluster_importance, width, label='Importance', alpha=0.7, color='#9467bd')
        ax2.bar(x + width/2, cluster_feasibility, width, label='Feasibility', alpha=0.7, color='#1f77b4')
        
        ax2.set_xlabel('Cluster', fontsize=12)
        ax2.set_ylabel('Mean Rating', fontsize=12)
        ax2.set_title('Cluster Characteristics - August 11 Dataset', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Cluster {i+1}\n(n={count})' for i, count in enumerate(cluster_counts)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.custom_dir}/cluster_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created improved cluster comparison")
    
    def create_improved_statement_frequency(self):
        """Create improved statement frequency analysis with better color scheme."""
        # Analyze grouping patterns
        grouping_counts = self.sorted_cards_df.groupby(['StatementID', 'PileID']).size().reset_index(name='Frequency')
        
        # Get top statements by frequency of being grouped together
        statement_freq = grouping_counts.groupby('StatementID')['Frequency'].sum().sort_values(ascending=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Top 20 most frequently grouped statements
        top_20 = statement_freq.head(20)
        bars1 = ax1.barh(range(len(top_20)), top_20.values, color='#9467bd', alpha=0.7)
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels([f"Statement {idx}" for idx in top_20.index])
        ax1.set_xlabel('Grouping Frequency')
        ax1.set_title('Top 20 Most Frequently Grouped Statements - August 11 Dataset', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(top_20.values):
            ax1.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
        # Distribution of grouping frequencies
        ax2.hist(statement_freq.values, bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax2.set_xlabel('Grouping Frequency')
        ax2.set_ylabel('Number of Statements')
        ax2.set_title('Distribution of Grouping Frequencies - August 11 Dataset', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_freq = statement_freq.mean()
        median_freq = statement_freq.median()
        ax2.axvline(mean_freq, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Mean: {mean_freq:.1f}')
        ax2.axvline(median_freq, color='#2ca02c', linestyle='--', linewidth=2, label=f'Median: {median_freq:.1f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.custom_dir}/statement_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created improved statement frequency analysis")
    
    def run_complete_analysis(self):
        """Run the complete analysis with all visualizations."""
        print("Running complete August 11 analysis with improved color scheme...")
        
        # Create all visualizations
        self.create_similarity_heatmap()
        self.create_optimal_clusters_analysis()
        self.create_improved_quadrant_analysis()
        self.create_improved_bubble_chart()
        self.create_improved_heatmap()
        self.create_improved_radar_chart()
        self.create_improved_cluster_comparison()
        self.create_improved_statement_frequency()
        
        print(f"\nComplete analysis finished!")
        print(f"Main analysis files saved to: {self.output_dir}/")
        print(f"Custom graphs saved to: {self.custom_dir}/")
        print("\nFiles created:")
        print("Main Analysis:")
        print("  - optimal_clusters_analysis.png (comprehensive cluster analysis)")
        print("  - cluster_summary.csv (cluster statistics)")
        print("  - cluster_ratings.csv (cluster assignments)")
        print("\nCustom Graphs:")
        print("  - similarity_heatmap.png (statement similarity matrix)")
        print("  - quadrant_analysis.png (improved quadrant analysis)")
        print("  - bubble_chart.png (improved bubble chart)")
        print("  - heatmap.png (improved importance/feasibility heatmaps)")
        print("  - radar_chart.png (improved radar chart)")
        print("  - cluster_comparison.png (improved cluster comparison)")
        print("  - statement_frequency.png (improved frequency analysis)")

if __name__ == "__main__":
    complete_analysis = August11CompleteAnalysis()
    complete_analysis.run_complete_analysis() 