import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import defaultdict
import warnings
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class August11ConceptMappingAnalyzer:
    def __init__(self):
        """Initialize the analyzer with August 11 data."""
        self.data_dir = "data/rcmap_august11_2025"
        self.output_dir = "Figures/august11_2025_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self.statements_df = pd.read_csv(f'{self.data_dir}/Statements.csv')
        self.ratings_df = pd.read_csv(f'{self.data_dir}/Ratings.csv')
        self.sorted_cards_df = pd.read_csv(f'{self.data_dir}/SortedCards.csv')
        self.demographics_df = pd.read_csv(f'{self.data_dir}/Demographics.csv')
        
        print(f"Loaded August 11 data:")
        print(f"  - {len(self.statements_df)} statements")
        print(f"  - {len(self.ratings_df)} ratings")
        print(f"  - {len(self.sorted_cards_df)} groupings")
        print(f"  - {len(self.demographics_df)} participants")
    
    def create_importance_feasibility_matrix(self):
        """Create importance vs feasibility matrix."""
        # Pivot data to get importance and feasibility for each statement
        importance_data = self.ratings_df[self.ratings_df['RatingType'] == 'Importance'].groupby('StatementID')['Rating'].agg(['mean', 'std', 'count']).reset_index()
        feasibility_data = self.ratings_df[self.ratings_df['RatingType'] == 'Feasibility'].groupby('StatementID')['Rating'].agg(['mean', 'std', 'count']).reset_index()
        
        # Merge with statements
        importance_data = importance_data.merge(self.statements_df, on='StatementID')
        feasibility_data = feasibility_data.merge(self.statements_df, on='StatementID')
        
        # Create combined matrix
        matrix_data = importance_data[['StatementID', 'StatementText', 'mean', 'std', 'count']].copy()
        matrix_data.columns = ['StatementID', 'StatementText', 'Importance_Mean', 'Importance_Std', 'Importance_Count']
        matrix_data = matrix_data.merge(feasibility_data[['StatementID', 'mean', 'std', 'count']], on='StatementID')
        matrix_data.columns = ['StatementID', 'StatementText', 'Importance_Mean', 'Importance_Std', 'Importance_Count', 'Feasibility_Mean', 'Feasibility_Std', 'Feasibility_Count']
        
        # Calculate gap (importance - feasibility)
        matrix_data['Gap'] = matrix_data['Importance_Mean'] - matrix_data['Feasibility_Mean']
        
        # Save to CSV
        matrix_data.to_csv(f'{self.output_dir}/importance_feasibility_summary.csv', index=False)
        
        return matrix_data
    
    def create_rating_distribution_plot(self):
        """Create rating distribution plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Importance ratings distribution
        importance_ratings = self.ratings_df[self.ratings_df['RatingType'] == 'Importance']['Rating']
        axes[0].hist(importance_ratings, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Distribution of Importance Ratings', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Importance Rating')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Feasibility ratings distribution
        feasibility_ratings = self.ratings_df[self.ratings_df['RatingType'] == 'Feasibility']['Rating']
        axes[1].hist(feasibility_ratings, bins=5, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_title('Distribution of Feasibility Ratings', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Feasibility Rating')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created rating distribution plot")
    
    def create_importance_vs_feasibility_plot(self):
        """Create importance vs feasibility scatter plot."""
        matrix_data = self.create_importance_feasibility_matrix()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create scatter plot
        scatter = ax.scatter(matrix_data['Feasibility_Mean'], matrix_data['Importance_Mean'], 
                           s=100, alpha=0.7, c=matrix_data['Gap'], cmap='RdYlBu_r')
        
        # Add diagonal line (importance = feasibility)
        min_val = min(matrix_data['Importance_Mean'].min(), matrix_data['Feasibility_Mean'].min())
        max_val = max(matrix_data['Importance_Mean'].max(), matrix_data['Feasibility_Mean'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Importance = Feasibility')
        
        # Add quadrant lines
        importance_median = matrix_data['Importance_Mean'].median()
        feasibility_median = matrix_data['Feasibility_Mean'].median()
        ax.axhline(y=importance_median, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=feasibility_median, color='gray', linestyle='-', alpha=0.3)
        
        # Add labels for some points
        for idx, row in matrix_data.iterrows():
            if row['Importance_Mean'] > importance_median + 0.5 or row['Feasibility_Mean'] > feasibility_median + 0.5:
                ax.annotate(f"{row['StatementID']}", 
                           (row['Feasibility_Mean'], row['Importance_Mean']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Customize plot
        ax.set_xlabel('Feasibility Rating (Mean)', fontsize=12)
        ax.set_ylabel('Importance Rating (Mean)', fontsize=12)
        ax.set_title('Importance vs Feasibility Ratings - August 11 Dataset', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Gap (Importance - Feasibility)', fontsize=10)
        
        # Add quadrant labels
        ax.text(0.05, 0.95, 'High Importance\nLow Feasibility', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.95, 0.95, 'High Importance\nHigh Feasibility', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax.text(0.05, 0.05, 'Low Importance\nLow Feasibility', transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.text(0.95, 0.05, 'Low Importance\nHigh Feasibility', transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/importance_vs_feasibility.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created importance vs feasibility plot")
    
    def perform_clustering_analysis(self):
        """Perform clustering analysis on the statements."""
        matrix_data = self.create_importance_feasibility_matrix()
        
        # Prepare data for clustering
        X = matrix_data[['Importance_Mean', 'Feasibility_Mean']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method
        wss = []
        silhouette_scores = []
        K_range = range(2, 11)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.plot(K_range, wss, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WSS)')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis for Optimal k')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/wss_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Choose optimal k (you can adjust this based on the plots)
        optimal_k = 4  # Based on typical concept mapping results
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        matrix_data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create cluster summary
        cluster_summary = matrix_data.groupby('Cluster').agg({
            'Importance_Mean': ['mean', 'std'],
            'Feasibility_Mean': ['mean', 'std'],
            'Gap': ['mean', 'std'],
            'StatementID': 'count'
        }).round(3)
        
        cluster_summary.columns = ['Importance_Mean', 'Importance_Std', 'Feasibility_Mean', 'Feasibility_Std', 'Gap_Mean', 'Gap_Std', 'Count']
        cluster_summary.to_csv(f'{self.output_dir}/cluster_summary.csv')
        
        # Save cluster assignments
        cluster_ratings = matrix_data[['StatementID', 'Cluster', 'Importance_Mean', 'Feasibility_Mean', 'Gap']]
        cluster_ratings.to_csv(f'{self.output_dir}/cluster_ratings.csv', index=False)
        
        print(f"Performed clustering analysis with {optimal_k} clusters")
        return matrix_data
    
    def create_concept_map(self):
        """Create a concept map visualization."""
        matrix_data = self.create_importance_feasibility_matrix()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (statements)
        for _, row in matrix_data.iterrows():
            G.add_node(row['StatementID'], 
                      importance=row['Importance_Mean'],
                      feasibility=row['Feasibility_Mean'],
                      gap=row['Gap'],
                      text=row['StatementText'][:50] + "..." if len(row['StatementText']) > 50 else row['StatementText'])
        
        # Add edges based on similarity (you can adjust the threshold)
        # For now, let's create a simple layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Draw nodes
        node_sizes = [G.nodes[node]['importance'] * 200 for node in G.nodes()]
        node_colors = [G.nodes[node]['gap'] for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, 
                              node_size=node_sizes,
                              node_color=node_colors,
                              cmap='RdYlBu_r',
                              alpha=0.8,
                              ax=ax)
        
        # Draw edges (if any)
        if G.edges():
            nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
        
        # Add labels
        labels = {node: f"{node}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Customize plot
        ax.set_title('Concept Map - August 11 Dataset\n(Node size = Importance, Color = Gap)', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r')
        sm.set_array(node_colors)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Gap (Importance - Feasibility)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/concept_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created concept map")
    
    def create_gap_analysis(self):
        """Create gap analysis visualization."""
        matrix_data = self.create_importance_feasibility_matrix()
        
        # Sort by gap
        matrix_data_sorted = matrix_data.sort_values('Gap', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Top 20 statements with largest gaps
        top_gaps = matrix_data_sorted.head(20)
        bars1 = ax1.barh(range(len(top_gaps)), top_gaps['Gap'], color='red', alpha=0.7)
        ax1.set_yticks(range(len(top_gaps)))
        ax1.set_yticklabels([f"{row['StatementID']}: {row['StatementText'][:40]}..." for _, row in top_gaps.iterrows()])
        ax1.set_xlabel('Gap (Importance - Feasibility)')
        ax1.set_title('Top 20 Statements with Largest Gaps (High Importance, Low Feasibility)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Bottom 20 statements with smallest gaps
        bottom_gaps = matrix_data_sorted.tail(20)
        bars2 = ax2.barh(range(len(bottom_gaps)), bottom_gaps['Gap'], color='green', alpha=0.7)
        ax2.set_yticks(range(len(bottom_gaps)))
        ax2.set_yticklabels([f"{row['StatementID']}: {row['StatementText'][:40]}..." for _, row in bottom_gaps.iterrows()])
        ax2.set_xlabel('Gap (Importance - Feasibility)')
        ax2.set_title('Bottom 20 Statements with Smallest Gaps (Low Importance, High Feasibility)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created gap analysis")
    
    def generate_html_report(self):
        """Generate an HTML report with all findings."""
        matrix_data = self.create_importance_feasibility_matrix()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>August 11 Concept Mapping Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>August 11 Concept Mapping Analysis Report</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report presents the concept mapping analysis for the August 11 BCCS AI Workshop dataset.</p>
                <ul>
                    <li><strong>Total Participants:</strong> {len(self.demographics_df)}</li>
                    <li><strong>Total Statements:</strong> {len(self.statements_df)}</li>
                    <li><strong>Total Ratings:</strong> {len(self.ratings_df)}</li>
                    <li><strong>Total Groupings:</strong> {len(self.sorted_cards_df)}</li>
                </ul>
            </div>
            
            <h2>Key Findings</h2>
            
            <h3>1. Rating Completion Rates</h3>
            <div class="highlight">
                <p><strong>Importance Rating Completion:</strong> {(len(self.ratings_df[self.ratings_df['RatingType'] == 'Importance']) / (len(self.demographics_df) * len(self.statements_df)) * 100):.1f}%</p>
                <p><strong>Feasibility Rating Completion:</strong> {(len(self.ratings_df[self.ratings_df['RatingType'] == 'Feasibility']) / (len(self.demographics_df) * len(self.statements_df)) * 100):.1f}%</p>
            </div>
            
            <h3>2. Top 10 Most Important Statements</h3>
            <table>
                <tr><th>Rank</th><th>Statement ID</th><th>Statement</th><th>Importance Mean</th><th>Feasibility Mean</th><th>Gap</th></tr>
        """
        
        # Add top 10 most important statements
        top_important = matrix_data.nlargest(10, 'Importance_Mean')
        for i, (_, row) in enumerate(top_important.iterrows(), 1):
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['StatementID']}</td>
                    <td>{row['StatementText']}</td>
                    <td>{row['Importance_Mean']:.2f}</td>
                    <td>{row['Feasibility_Mean']:.2f}</td>
                    <td>{row['Gap']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h3>3. Top 10 Most Feasible Statements</h3>
            <table>
                <tr><th>Rank</th><th>Statement ID</th><th>Statement</th><th>Importance Mean</th><th>Feasibility Mean</th><th>Gap</th></tr>
        """
        
        # Add top 10 most feasible statements
        top_feasible = matrix_data.nlargest(10, 'Feasibility_Mean')
        for i, (_, row) in enumerate(top_feasible.iterrows(), 1):
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['StatementID']}</td>
                    <td>{row['StatementText']}</td>
                    <td>{row['Importance_Mean']:.2f}</td>
                    <td>{row['Feasibility_Mean']:.2f}</td>
                    <td>{row['Gap']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h3>4. Priority Action Items (High Importance, Low Feasibility)</h3>
            <table>
                <tr><th>Rank</th><th>Statement ID</th><th>Statement</th><th>Importance Mean</th><th>Feasibility Mean</th><th>Gap</th></tr>
        """
        
        # Add top 10 priority items (high gap)
        top_gaps = matrix_data.nlargest(10, 'Gap')
        for i, (_, row) in enumerate(top_gaps.iterrows(), 1):
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['StatementID']}</td>
                    <td>{row['StatementText']}</td>
                    <td>{row['Importance_Mean']:.2f}</td>
                    <td>{row['Feasibility_Mean']:.2f}</td>
                    <td>{row['Gap']:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Visualizations</h2>
            <p>The following visualizations provide additional insights into the concept mapping results:</p>
            
            <h3>Importance vs Feasibility Plot</h3>
            <img src="importance_vs_feasibility.png" alt="Importance vs Feasibility Plot">
            
            <h3>Rating Distribution</h3>
            <img src="rating_distribution.png" alt="Rating Distribution">
            
            <h3>Concept Map</h3>
            <img src="concept_map.png" alt="Concept Map">
            
            <h3>Gap Analysis</h3>
            <img src="gap_analysis.png" alt="Gap Analysis">
            
            <h2>Recommendations</h2>
            <div class="highlight">
                <h3>Immediate Actions (High Importance, High Feasibility)</h3>
                <p>Focus on implementing statements that are both highly important and highly feasible.</p>
                
                <h3>Strategic Planning (High Importance, Low Feasibility)</h3>
                <p>Develop long-term strategies for addressing statements with high importance but low feasibility.</p>
                
                <h3>Resource Optimization (Low Importance, High Feasibility)</h3>
                <p>Consider whether resources should be allocated to easily implementable but less important items.</p>
            </div>
            
            <p><em>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}/analysis_report.html', 'w') as f:
            f.write(html_content)
        
        print("Generated HTML report")
    
    def run_complete_analysis(self):
        """Run the complete concept mapping analysis."""
        print("Starting August 11 Concept Mapping Analysis...")
        
        # Create all visualizations and analyses
        self.create_rating_distribution_plot()
        self.create_importance_vs_feasibility_plot()
        self.perform_clustering_analysis()
        self.create_concept_map()
        self.create_gap_analysis()
        self.generate_html_report()
        
        print(f"\nAnalysis complete! All files saved to {self.output_dir}/")
        print("Files created:")
        print("  - analysis_report.html (comprehensive report)")
        print("  - importance_feasibility_summary.csv (data summary)")
        print("  - cluster_summary.csv (clustering results)")
        print("  - importance_vs_feasibility.png (scatter plot)")
        print("  - rating_distribution.png (histograms)")
        print("  - concept_map.png (network visualization)")
        print("  - gap_analysis.png (priority analysis)")
        print("  - wss_plot.png (clustering analysis)")

if __name__ == "__main__":
    analyzer = August11ConceptMappingAnalyzer()
    analyzer.run_complete_analysis() 