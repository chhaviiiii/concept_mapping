import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict
import warnings
import os
from matplotlib.patches import Circle
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class August11CustomGraphs:
    def __init__(self):
        """Initialize the custom graphs generator for August 11 data."""
        self.data_dir = "data/rcmap_august11_2025"
        self.output_dir = "Figures/custom_graphs_august11"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.statements_df = pd.read_csv(f'{self.data_dir}/Statements.csv')
        self.ratings_df = pd.read_csv(f'{self.data_dir}/Ratings.csv')
        self.sorted_cards_df = pd.read_csv(f'{self.data_dir}/SortedCards.csv')
        self.demographics_df = pd.read_csv(f'{self.data_dir}/Demographics.csv')
        
        # Create importance-feasibility matrix
        self.create_matrix()
        
        print(f"Loaded August 11 data for custom graphs:")
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
    
    def create_quadrant_analysis(self):
        """Create detailed quadrant analysis visualization."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Calculate medians for quadrant boundaries
        importance_median = self.matrix_data['Importance_Mean'].median()
        feasibility_median = self.matrix_data['Feasibility_Mean'].median()
        
        # Define quadrants
        quadrants = {
            'Q1': {'name': 'High Importance, High Feasibility', 'color': 'green', 'alpha': 0.3},
            'Q2': {'name': 'High Importance, Low Feasibility', 'color': 'red', 'alpha': 0.3},
            'Q3': {'name': 'Low Importance, Low Feasibility', 'color': 'gray', 'alpha': 0.3},
            'Q4': {'name': 'Low Importance, High Feasibility', 'color': 'blue', 'alpha': 0.3}
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
                      alpha=0.7, 
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
        
        # Add quadrant labels
        ax.text(0.95, 0.95, 'Q1: High Importance\nHigh Feasibility\n(Immediate Actions)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.text(0.05, 0.95, 'Q2: High Importance\nLow Feasibility\n(Strategic Planning)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.text(0.05, 0.05, 'Q3: Low Importance\nLow Feasibility\n(Monitor)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.text(0.95, 0.05, 'Q4: Low Importance\nHigh Feasibility\n(Quick Wins)', 
                transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
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
                horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/quadrant_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created quadrant analysis")
    
    def create_bubble_chart(self):
        """Create bubble chart with importance, feasibility, and gap."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create bubble chart
        scatter = ax.scatter(self.matrix_data['Feasibility_Mean'], 
                           self.matrix_data['Importance_Mean'],
                           s=self.matrix_data['Importance_Mean'] * 100,  # Size based on importance
                           c=self.matrix_data['Gap'],  # Color based on gap
                           cmap='RdYlBu_r',
                           alpha=0.7,
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
        plt.savefig(f'{self.output_dir}/bubble_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created bubble chart")
    
    def create_heatmap(self):
        """Create heatmap of importance vs feasibility ratings."""
        # Create pivot table for heatmap
        importance_pivot = self.ratings_df[self.ratings_df['RatingType'] == 'Importance'].pivot_table(
            index='StatementID', columns='ParticipantID', values='Rating', aggfunc='mean'
        )
        
        feasibility_pivot = self.ratings_df[self.ratings_df['RatingType'] == 'Feasibility'].pivot_table(
            index='StatementID', columns='ParticipantID', values='Rating', aggfunc='mean'
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Importance heatmap
        sns.heatmap(importance_pivot, ax=ax1, cmap='YlOrRd', cbar_kws={'label': 'Importance Rating'})
        ax1.set_title('Importance Ratings Heatmap - August 11', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Participant ID')
        ax1.set_ylabel('Statement ID')
        
        # Feasibility heatmap
        sns.heatmap(feasibility_pivot, ax=ax2, cmap='Blues', cbar_kws={'label': 'Feasibility Rating'})
        ax2.set_title('Feasibility Ratings Heatmap - August 11', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Participant ID')
        ax2.set_ylabel('Statement ID')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created heatmap")
    
    def create_radar_chart(self):
        """Create radar chart for top statements."""
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
        ax.plot(angles, importance_values_complete, 'o-', linewidth=2, label='Importance', color='red')
        ax.fill(angles, importance_values_complete, alpha=0.25, color='red')
        
        # Plot feasibility values - ensure arrays have same length
        feasibility_values_complete = np.concatenate([feasibility_values, [feasibility_values[0]]])
        ax.plot(angles, feasibility_values_complete, 'o-', linewidth=2, label='Feasibility', color='blue')
        ax.fill(angles, feasibility_values_complete, alpha=0.25, color='blue')
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_title('Top 5 Most Important Statements - August 11 Dataset', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created radar chart")
    
    def create_cluster_comparison(self):
        """Create cluster comparison visualization."""
        # Perform clustering
        X = self.matrix_data[['Importance_Mean', 'Feasibility_Mean']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        self.matrix_data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create cluster comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Scatter plot with clusters
        colors = ['red', 'blue', 'green', 'orange']
        for i in range(4):
            cluster_data = self.matrix_data[self.matrix_data['Cluster'] == i]
            ax1.scatter(cluster_data['Feasibility_Mean'], cluster_data['Importance_Mean'],
                       c=colors[i], label=f'Cluster {i+1}', alpha=0.7, s=100)
        
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
        
        ax2.bar(x - width/2, cluster_importance, width, label='Importance', alpha=0.7)
        ax2.bar(x + width/2, cluster_feasibility, width, label='Feasibility', alpha=0.7)
        
        ax2.set_xlabel('Cluster', fontsize=12)
        ax2.set_ylabel('Mean Rating', fontsize=12)
        ax2.set_title('Cluster Characteristics - August 11 Dataset', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Cluster {i+1}\n(n={count})' for i, count in enumerate(cluster_counts)])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cluster_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created cluster comparison")
    
    def create_statement_frequency(self):
        """Create statement frequency analysis."""
        # Analyze grouping patterns
        grouping_counts = self.sorted_cards_df.groupby(['StatementID', 'PileID']).size().reset_index(name='Frequency')
        
        # Get top statements by frequency of being grouped together
        statement_freq = grouping_counts.groupby('StatementID')['Frequency'].sum().sort_values(ascending=False)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Top 20 most frequently grouped statements
        top_20 = statement_freq.head(20)
        bars1 = ax1.barh(range(len(top_20)), top_20.values, color='skyblue', alpha=0.7)
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels([f"Statement {idx}" for idx in top_20.index])
        ax1.set_xlabel('Grouping Frequency')
        ax1.set_title('Top 20 Most Frequently Grouped Statements - August 11 Dataset', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(top_20.values):
            ax1.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
        # Distribution of grouping frequencies
        ax2.hist(statement_freq.values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Grouping Frequency')
        ax2.set_ylabel('Number of Statements')
        ax2.set_title('Distribution of Grouping Frequencies - August 11 Dataset', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_freq = statement_freq.mean()
        median_freq = statement_freq.median()
        ax2.axvline(mean_freq, color='red', linestyle='--', label=f'Mean: {mean_freq:.1f}')
        ax2.axvline(median_freq, color='green', linestyle='--', label=f'Median: {median_freq:.1f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/statement_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created statement frequency analysis")
    
    def run_all_custom_graphs(self):
        """Run all custom graph generation."""
        print("Generating custom graphs for August 11 dataset...")
        
        self.create_quadrant_analysis()
        self.create_bubble_chart()
        self.create_heatmap()
        self.create_radar_chart()
        self.create_cluster_comparison()
        self.create_statement_frequency()
        
        print(f"\nAll custom graphs created and saved to {self.output_dir}/")
        print("Files created:")
        print("  - quadrant_analysis.png (detailed quadrant analysis)")
        print("  - bubble_chart.png (bubble chart with importance, feasibility, gap)")
        print("  - heatmap.png (importance and feasibility heatmaps)")
        print("  - radar_chart.png (radar chart for top statements)")
        print("  - cluster_comparison.png (clustering analysis)")
        print("  - statement_frequency.png (grouping frequency analysis)")

if __name__ == "__main__":
    custom_graphs = August11CustomGraphs()
    custom_graphs.run_all_custom_graphs() 