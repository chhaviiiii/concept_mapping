"""
Visualization functionality for concept mapping analysis.

This module provides comprehensive visualization capabilities for concept mapping,
including point maps, cluster maps, rating maps, and various statistical plots.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial import ConvexHull
from sklearn.metrics import silhouette_score
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ConceptMapVisualizer:
    """
    Handles all visualization tasks for concept mapping analysis.
    
    This class provides methods to create various types of plots including:
    - Point maps (MDS configuration)
    - Cluster maps with convex hulls
    - Point rating maps
    - Cluster rating maps
    - Pattern matching plots
    - Go-zone plots
    - Dendrograms
    - Parallel coordinate plots
    """
    
    def __init__(self, output_folder: Path):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        output_folder : Path
            Path to the output folder for saving plots
        """
        self.output_folder = output_folder
        self.output_folder.mkdir(exist_ok=True)
        
        # Create figures subdirectory
        self.figures_folder = self.output_folder / 'figures'
        self.figures_folder.mkdir(exist_ok=True)
        
        # Set default parameters
        self.figsize = (10, 8)
        self.dpi = 300
        self.bbox_inches = 'tight'
        
        # Color schemes
        self.color_schemes = {
            'default': 'husl',
            'viridis': 'viridis',
            'plasma': 'plasma',
            'coolwarm': 'coolwarm'
        }
        
        self.current_scheme = 'default'
    
    def create_point_map(self, mds_coords: np.ndarray, 
                        statements: pd.DataFrame,
                        save: bool = True) -> plt.Figure:
        """
        Create a point map showing MDS configuration.
        
        Parameters
        ----------
        mds_coords : np.ndarray
            MDS coordinates (n_statements, 2)
        statements : pd.DataFrame
            Statements data
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot points with vibrant blue
        ax.scatter(mds_coords[:, 0], mds_coords[:, 1], 
                  c='#2196F3', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add statement numbers
        for i, (x, y) in enumerate(mds_coords):
            ax.annotate(str(i + 1), (x, y), xytext=(2, 2), 
                       textcoords='offset points', fontsize=8, ha='left')
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('Figure 1a: Point Map (MDS Configuration)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save:
            plt.savefig(self.figures_folder / 'figure_1a_point_map.png', 
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_cluster_map(self, mds_coords: np.ndarray,
                          cluster_labels: np.ndarray,
                          statements: pd.DataFrame,
                          save: bool = True) -> plt.Figure:
        """
        Create a cluster map with convex hulls.
        
        Parameters
        ----------
        mds_coords : np.ndarray
            MDS coordinates
        cluster_labels : np.ndarray
            Cluster assignments
        statements : pd.DataFrame
            Statements data
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_clusters = len(np.unique(cluster_labels))
        # Use darker, more saturated colors
        dark_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
                      '#BC4749', '#7209B7', '#3A86FF', '#FF006E', '#8338EC']
        colors = [dark_colors[i % len(dark_colors)] for i in range(n_clusters)]
        
        # Plot points by cluster
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_coords = mds_coords[mask]
            
            ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                      c=[colors[cluster_id]], s=50, alpha=0.7,
                      edgecolors='black', linewidth=0.5,
                      label=f'Cluster {cluster_id + 1}')
            
            # Add convex hull
            if len(cluster_coords) > 2:
                try:
                    hull = ConvexHull(cluster_coords)
                    hull_points = cluster_coords[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the hull
                    
                    ax.plot(hull_points[:, 0], hull_points[:, 1], 
                           color=colors[cluster_id], linewidth=2, alpha=0.8)
                except:
                    pass  # Skip hull if not enough points
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('Figure 1b: Cluster Map', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save:
            plt.savefig(self.figures_folder / 'figure_1b_cluster_map.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_combined_figure_1(self, mds_coords: np.ndarray,
                                cluster_labels: np.ndarray,
                                statements: pd.DataFrame,
                                save: bool = True) -> plt.Figure:
        """
        Create a combined figure showing both point map (1a) and cluster map (1b) side-by-side.
        
        Parameters
        ----------
        mds_coords : np.ndarray
            MDS coordinates (n_statements, 2)
        cluster_labels : np.ndarray
            Cluster assignments
        statements : pd.DataFrame
            Statements data
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        # Create figure with two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left subplot: Point Map (Figure 1a)
        ax1.scatter(mds_coords[:, 0], mds_coords[:, 1], 
                   c='#2196F3', s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add statement numbers
        for i, (x, y) in enumerate(mds_coords):
            ax1.annotate(str(i + 1), (x, y), xytext=(2, 2), 
                        textcoords='offset points', fontsize=8, ha='left')
        
        ax1.set_xlabel('Dimension 1', fontsize=12)
        ax1.set_ylabel('Dimension 2', fontsize=12)
        ax1.set_title('Figure 1a: Point Map (MDS Configuration)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Right subplot: Cluster Map (Figure 1b)
        n_clusters = len(np.unique(cluster_labels))
        # Use darker, more saturated colors
        dark_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
                      '#BC4749', '#7209B7', '#3A86FF', '#FF006E', '#8338EC']
        colors = [dark_colors[i % len(dark_colors)] for i in range(n_clusters)]
        
        # Plot points by cluster
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_coords = mds_coords[mask]
            
            ax2.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                       c=[colors[cluster_id]], s=50, alpha=0.7,
                       edgecolors='black', linewidth=0.5,
                       label=f'Cluster {cluster_id + 1}')
            
            # Add convex hull
            if len(cluster_coords) > 2:
                try:
                    hull = ConvexHull(cluster_coords)
                    hull_points = cluster_coords[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the hull
                    
                    ax2.plot(hull_points[:, 0], hull_points[:, 1], 
                            color=colors[cluster_id], linewidth=2, alpha=0.8)
                except:
                    pass  # Skip hull if not enough points
        
        ax2.set_xlabel('Dimension 1', fontsize=12)
        ax2.set_ylabel('Dimension 2', fontsize=12)
        ax2.set_title('Figure 1b: Cluster Map', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Adjust layout
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_folder / 'figure_1_combined.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_point_rating_map(self, mds_coords: np.ndarray,
                               statement_summary: pd.DataFrame,
                               rating_var: str = 'Importance',
                               save: bool = True) -> plt.Figure:
        """
        Create a point rating map with point sizes based on ratings.
        
        Parameters
        ----------
        mds_coords : np.ndarray
            MDS coordinates
        statement_summary : pd.DataFrame
            Statement summary statistics
        rating_var : str
            Rating variable to use for point sizes
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Check for rating columns (they end with _mean)
        rating_col = f'{rating_var}_mean'
        if rating_col in statement_summary.columns:
            # Get ratings for both size and color
            ratings = statement_summary[rating_col].values
            min_rating = ratings.min()
            max_rating = ratings.max()
            
            if max_rating > min_rating:
                # Normalize ratings for point sizes
                normalized_ratings = (ratings - min_rating) / (max_rating - min_rating)
                point_sizes = 50 + normalized_ratings * 200  # 50-250 range
            else:
                point_sizes = np.full(len(ratings), 100)  # Default size
                normalized_ratings = np.zeros(len(ratings))
        else:
            ratings = np.full(len(mds_coords), 3.0)  # Default rating
            point_sizes = np.full(len(mds_coords), 100)
            min_rating = 2.0
            max_rating = 4.0
            normalized_ratings = np.zeros(len(mds_coords))
        
        # Use a colormap with vibrant colors
        # Purple → blue → teal → green → yellow
        colors_list = ['#9C27B0', '#2196F3', '#00BCD4', '#4CAF50', '#FFEB3B']  # Vibrant gradient
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('importance', colors_list, N=n_bins)
        
        # Plot points with both size and color representing importance
        scatter = ax.scatter(mds_coords[:, 0], mds_coords[:, 1],
                           s=point_sizes, c=ratings, cmap=cmap,
                           vmin=min_rating, vmax=max_rating,
                           alpha=0.9, edgecolors='black', linewidth=0.5)
        
        # Add colorbar on the right
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(f'{rating_var} Rating', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'Size and Color = {rating_var} Rating', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save:
            plt.savefig(self.figures_folder / 'figure_2_point_rating_map.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_cluster_rating_map(self, cluster_means: pd.DataFrame,
                                 save: bool = True) -> plt.Figure:
        """
        Create a cluster rating map showing cluster-level statistics.
        
        Parameters
        ----------
        cluster_means : pd.DataFrame
            Cluster-level statistics
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_clusters = len(cluster_means)
        # Use darker, more saturated colors
        dark_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
                      '#BC4749', '#7209B7', '#3A86FF', '#FF006E', '#8338EC']
        colors = [dark_colors[i % len(dark_colors)] for i in range(n_clusters)]
        
        # Create cluster positions
        cluster_positions = np.linspace(-1.5, 1.5, n_clusters)
        
        for i, (cluster_id, row) in enumerate(cluster_means.iterrows()):
            x_pos = cluster_positions[i]
            
            # Create irregular quadrilateral
            base_size = 0.6 + (row.get('Importance', 3) - 2) / 2 * 0.4
            
            # Define quadrilateral points
            quad_points = np.array([
                [x_pos - base_size*0.8, -0.4],
                [x_pos + base_size*1.2, -0.3],
                [x_pos + base_size*0.9, 0.4],
                [x_pos - base_size*0.6, 0.5]
            ])
            
            # Add randomness
            np.random.seed(i)
            random_offset = np.random.normal(0, 0.05, quad_points.shape)
            quad_points += random_offset
            
            # Create the quadrilateral
            quad = plt.Polygon(quad_points, facecolor=colors[i], 
                            edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(quad)
            
            # Add text inside
            quad_center_x = np.mean(quad_points[:, 0])
            quad_center_y = np.mean(quad_points[:, 1])
            
            ax.text(quad_center_x, quad_center_y + 0.15, f'Cluster {cluster_id+1}',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            
            importance = row.get('Importance', 0)
            ax.text(quad_center_x, quad_center_y - 0.15, f'{importance:.2f}',
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.axis('off')
        ax.set_title('Figure 3: Cluster Rating Map', fontsize=16, fontweight='bold')
        
        if save:
            plt.savefig(self.figures_folder / 'figure_3_cluster_rating_map.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_pattern_match(self, cluster_means: pd.DataFrame,
                           save: bool = True) -> plt.Figure:
        """
        Create a pattern match plot comparing cluster ratings.
        
        Parameters
        ----------
        cluster_means : pd.DataFrame
            Cluster-level statistics
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Importance', 'Feasibility']
        x_positions = [1, 2]
        # Use darker, more saturated colors
        dark_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
                      '#BC4749', '#7209B7', '#3A86FF', '#FF006E', '#8338EC']
        colors = [dark_colors[i % len(dark_colors)] for i in range(len(cluster_means))]
        
        for i, (cluster_id, row) in enumerate(cluster_means.iterrows()):
            imp_mean = row.get('Importance', 0)
            feas_mean = row.get('Feasibility', 0)
            
            ax.plot(x_positions, [imp_mean, feas_mean], 'o-', 
                   linewidth=3, color=colors[i], 
                   label=f'Cluster {cluster_id+1}', markersize=10)
        
        # Calculate correlation
        if 'Importance' in cluster_means.columns and 'Feasibility' in cluster_means.columns:
            correlation = np.corrcoef(cluster_means['Importance'], 
                                    cluster_means['Feasibility'])[0, 1]
            
            max_rating = max(cluster_means['Importance'].max(), 
                           cluster_means['Feasibility'].max())
            text_y = max_rating + 0.2
            
            ax.text(1.5, text_y, f'Correlation (r) = {correlation:.3f}',
                   fontsize=12, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories)
        ax.set_title('Figure 4: Pattern Match Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Rating')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.figures_folder / 'figure_4_pattern_match.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_go_zone_plot(self, statement_summary: pd.DataFrame,
                          save: bool = True) -> plt.Figure:
        """
        Create a go-zone plot showing importance vs feasibility.
        
        Parameters
        ----------
        statement_summary : pd.DataFrame
            Statement summary statistics
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Check for rating columns (they end with _mean)
        imp_col = 'Importance_mean'
        feas_col = 'Feasibility_mean'
        
        if imp_col not in statement_summary.columns or feas_col not in statement_summary.columns:
            print("Warning: Importance_mean or Feasibility_mean columns not found")
            return fig
        
        # Calculate means
        imp_mean = statement_summary[imp_col].mean()
        feas_mean = statement_summary[feas_col].mean()
        
        # Define vibrant quadrant colors
        # Top-right: High Imp, High Feas - Green
        # Bottom-right: High Imp, Low Feas - Purple
        # Top-left: Low Imp, High Feas - Orange
        # Bottom-left: Low Imp, Low Feas - Red
        colors = {
            'high_imp_high_feas': '#4CAF50',      # Vibrant green
            'high_imp_low_feas': '#9C27B0',       # Vibrant purple
            'low_imp_high_feas': '#FF9800',       # Vibrant orange
            'low_imp_low_feas': '#F44336'         # Vibrant red
        }
        
        # Assign colors based on quadrants
        point_colors = []
        for idx, row in statement_summary.iterrows():
            imp_val = row[imp_col]
            feas_val = row[feas_col]
            
            if imp_val >= imp_mean and feas_val >= feas_mean:
                # Top-right: High Imp, High Feas
                point_colors.append(colors['high_imp_high_feas'])
            elif imp_val >= imp_mean and feas_val < feas_mean:
                # Bottom-right: High Imp, Low Feas
                point_colors.append(colors['high_imp_low_feas'])
            elif imp_val < imp_mean and feas_val >= feas_mean:
                # Top-left: Low Imp, High Feas
                point_colors.append(colors['low_imp_high_feas'])
            else:
                # Bottom-left: Low Imp, Low Feas
                point_colors.append(colors['low_imp_low_feas'])
        
        # Create scatter plot with quadrant-based colors
        scatter = ax.scatter(statement_summary[feas_col], 
                           statement_summary[imp_col],
                           c=point_colors, s=50, alpha=0.8,
                           edgecolors='black', linewidth=0.5)
        
        # Add quadrant lines
        ax.axhline(y=imp_mean, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axvline(x=feas_mean, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add mean labels on axes
        ax.text(feas_mean, ax.get_ylim()[1] * 0.98, f'Mean Feasibility ({feas_mean:.2f})',
               ha='center', va='top', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.text(ax.get_xlim()[0] * 1.02, imp_mean, f'Mean Importance ({imp_mean:.2f})',
               ha='left', va='center', fontsize=9, rotation=90,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add quadrant labels with darker colors
        ax.text(0.02, 0.98, 'High Imp, Low Feas', 
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor=colors['high_imp_low_feas'], 
                        alpha=0.7, edgecolor='black'))
        
        ax.text(0.98, 0.98, 'High Imp, High Feas', 
               transform=ax.transAxes, fontsize=10, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor=colors['high_imp_high_feas'], 
                        alpha=0.7, edgecolor='black'))
        
        ax.text(0.02, 0.02, 'Low Imp, Low Feas', 
               transform=ax.transAxes, fontsize=10, va='bottom',
               bbox=dict(boxstyle='round', facecolor=colors['low_imp_low_feas'], 
                        alpha=0.7, edgecolor='black'))
        
        ax.text(0.98, 0.02, 'Low Imp, High Feas', 
               transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor=colors['low_imp_high_feas'], 
                        alpha=0.7, edgecolor='black'))
        
        ax.set_xlabel('Feasibility', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Figure 5: Go-Zone Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.figures_folder / 'figure_5_go_zone_plot.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_dendrogram(self, mds_coords: np.ndarray,
                        cluster_labels: np.ndarray,
                        save: bool = True) -> plt.Figure:
        """
        Create a dendrogram showing cluster hierarchy.
        
        Parameters
        ----------
        mds_coords : np.ndarray
            MDS coordinates
        cluster_labels : np.ndarray
            Cluster assignments
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create linkage matrix
        linkage_matrix = linkage(mds_coords, method='ward')
        
        # Create dendrogram
        dendrogram(linkage_matrix, ax=ax, color_threshold=0.7*max(linkage_matrix[:, 2]))
        
        ax.set_title('Cluster Dendrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Statement Index', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        if save:
            plt.savefig(self.figures_folder / 'dendrogram.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_parallel_coordinates(self, cluster_means: pd.DataFrame,
                                   save: bool = True) -> plt.Figure:
        """
        Create a parallel coordinates plot.
        
        Parameters
        ----------
        cluster_means : pd.DataFrame
            Cluster-level statistics
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get numeric columns
        numeric_cols = cluster_means.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Warning: Not enough numeric columns for parallel coordinates")
            return fig
        
        # Normalize data
        normalized_data = cluster_means[numeric_cols].copy()
        for col in numeric_cols:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        # Use darker, more saturated colors
        dark_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', 
                      '#BC4749', '#7209B7', '#3A86FF', '#FF006E', '#8338EC']
        
        # Plot parallel coordinates
        for i, (cluster_id, row) in enumerate(normalized_data.iterrows()):
            color = dark_colors[i % len(dark_colors)]
            ax.plot(range(len(numeric_cols)), row.values, 
                   marker='o', linewidth=2.5, label=f'Cluster {cluster_id+1}', 
                   color=color, alpha=0.9)
        
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45)
        ax.set_ylabel('Normalized Rating', fontsize=12)
        ax.set_title('Parallel Coordinates Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.figures_folder / 'parallel_coordinates.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def set_color_scheme(self, scheme: str):
        """
        Set the color scheme for plots.
        
        Parameters
        ----------
        scheme : str
            Color scheme name ('default', 'viridis', 'plasma', 'coolwarm')
        """
        if scheme in self.color_schemes:
            self.current_scheme = scheme
            sns.set_palette(self.color_schemes[scheme])
        else:
            warnings.warn(f"Unknown color scheme: {scheme}")
    
    def create_subgroup_comparison_plot(self, subgroup_results: Dict,
                                       save: bool = True) -> plt.Figure:
        """
        Create a plot comparing ratings across subgroups for each cluster.
        
        Parameters
        ----------
        subgroup_results : Dict
            Results from subgroup analysis
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        if not subgroup_results or 'results' not in subgroup_results:
            print("Warning: No subgroup results to plot")
            return None
        
        demographic_var = subgroup_results.get('demographic_var', 'Subgroup')
        results = subgroup_results['results']
        
        # Get all clusters and rating variables
        clusters = sorted(results.keys())
        if not clusters:
            return None
        
        # Get rating variables from first cluster
        first_cluster = clusters[0]
        rating_vars = list(results[first_cluster].keys())
        
        # Create subplots: one for each rating variable
        n_rating_vars = len(rating_vars)
        fig, axes = plt.subplots(1, n_rating_vars, figsize=(6 * n_rating_vars, 8))
        
        if n_rating_vars == 1:
            axes = [axes]
        
        # Use vibrant colors for subgroups
        vibrant_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', 
                         '#00BCD4', '#FFC107', '#E91E63', '#3F51B5', '#009688']
        
        for ax_idx, rating_var in enumerate(rating_vars):
            ax = axes[ax_idx]
            
            # Prepare data for plotting
            cluster_ids = []
            subgroup_means = {}
            subgroup_stds = {}
            subgroup_labels = []
            
            for cluster_key in clusters:
                cluster_id = int(cluster_key.split('_')[1])
                cluster_ids.append(cluster_id)
                
                if rating_var in results[cluster_key]:
                    cluster_data = results[cluster_key][rating_var]
                    stats = cluster_data['subgroup_stats']
                    subgroups = cluster_data['subgroups']
                    
                    for i, subgroup in enumerate(subgroups):
                        if subgroup not in subgroup_means:
                            subgroup_means[subgroup] = []
                            subgroup_stds[subgroup] = []
                            subgroup_labels.append(subgroup)
                        
                        subgroup_means[subgroup].append(stats[subgroup]['mean'])
                        subgroup_stds[subgroup].append(stats[subgroup]['std'])
            
            # Plot grouped bar chart
            x = np.arange(len(cluster_ids))
            width = 0.8 / len(subgroup_labels)
            
            for i, subgroup in enumerate(subgroup_labels):
                means = subgroup_means[subgroup]
                stds = subgroup_stds[subgroup]
                color = vibrant_colors[i % len(vibrant_colors)]
                
                offset = (i - len(subgroup_labels) / 2 + 0.5) * width
                ax.bar(x + offset, means, width, yerr=stds, 
                      label=subgroup, color=color, alpha=0.8, 
                      capsize=5, edgecolor='black', linewidth=0.5)
            
            # Add significance markers
            for i, cluster_key in enumerate(clusters):
                if rating_var in results[cluster_key]:
                    test_results = results[cluster_key][rating_var]['test_results']
                    if test_results and 'p_value' in test_results:
                        p_val = test_results['p_value']
                        if p_val < 0.001:
                            sig_text = '***'
                        elif p_val < 0.01:
                            sig_text = '**'
                        elif p_val < 0.05:
                            sig_text = '*'
                        else:
                            sig_text = 'ns'
                        
                        # Add text above bars
                        max_mean = max([subgroup_means[sub][i] for sub in subgroup_labels])
                        ax.text(i, max_mean + 0.1, sig_text, 
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel(f'{rating_var} Rating', fontsize=12)
            ax.set_title(f'{rating_var} by {demographic_var}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Cluster {cid}' for cid in cluster_ids])
            ax.legend(title=demographic_var, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_folder / f'subgroup_comparison_{demographic_var.lower()}.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_subgroup_violin_plots(self, ratings_with_demo: pd.DataFrame,
                                     demographic_var: str,
                                     save: bool = True) -> plt.Figure:
        """
        Create violin plots with swarm plots for subgroup comparisons.
        
        Parameters
        ----------
        ratings_with_demo : pd.DataFrame
            Ratings data merged with demographics and cluster info
        demographic_var : str
            Demographic variable name
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        rating_vars = [col for col in ratings_with_demo.columns 
                      if col not in ['RaterID', 'StatementID', 'Cluster', demographic_var, 'Statement']]
        
        n_vars = len(rating_vars)
        n_clusters = ratings_with_demo['Cluster'].nunique()
        
        fig, axes = plt.subplots(n_clusters, n_vars, 
                                figsize=(6 * n_vars, 5 * n_clusters))
        
        if n_clusters == 1:
            axes = axes.reshape(1, -1)
        if n_vars == 1:
            axes = axes.reshape(-1, 1)
        
        vibrant_colors = ['#2196F3', '#4CAF50']
        
        for cluster_idx, cluster_id in enumerate(sorted(ratings_with_demo['Cluster'].unique())):
            cluster_data = ratings_with_demo[ratings_with_demo['Cluster'] == cluster_id]
            
            for var_idx, rating_var in enumerate(rating_vars):
                ax = axes[cluster_idx, var_idx]
                
                subgroups = sorted(cluster_data[demographic_var].dropna().unique())
                
                data_to_plot = []
                labels = []
                colors_list = []
                
                for i, subgroup in enumerate(subgroups):
                    subgroup_data = cluster_data[
                        (cluster_data[demographic_var] == subgroup) & 
                        (cluster_data[rating_var].notna())
                    ][rating_var].values
                    
                    if len(subgroup_data) > 0:
                        data_to_plot.append(subgroup_data)
                        labels.append(subgroup)
                        colors_list.append(vibrant_colors[i % len(vibrant_colors)])
                
                # Create violin plot
                parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                                     showmeans=True, showmedians=True)
                
                # Color violins
                for pc, color in zip(parts['bodies'], colors_list):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                # Add swarm plot overlay
                for i, data in enumerate(data_to_plot):
                    x_pos = np.random.normal(i, 0.05, len(data))
                    ax.scatter(x_pos, data, alpha=0.3, s=10, color='black', zorder=3)
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_ylabel(f'{rating_var} Rating', fontsize=11)
                ax.set_title(f'Cluster {cluster_id}: {rating_var}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_folder / f'subgroup_violin_{demographic_var.lower()}.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def create_subgroup_pattern_match(self, subgroup_enhanced_results: Dict,
                                     save: bool = True) -> plt.Figure:
        """
        Create two-bar pattern match plot (clinical vs non-clinical) for each cluster.
        Includes rating values on bars, p-values, and Hedge's g with interpretation.
        
        Parameters
        ----------
        subgroup_enhanced_results : Dict
            Enhanced subgroup analysis results
        save : bool
            Whether to save the plot
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        if not subgroup_enhanced_results or 'cluster_table' not in subgroup_enhanced_results:
            return None
        
        cluster_df = subgroup_enhanced_results['cluster_table']
        
        # Get unique clusters and metrics
        clusters = sorted(cluster_df['Cluster'].unique())
        metrics = sorted(cluster_df['Metric'].unique())
        
        n_clusters = len(clusters)
        n_metrics = len(metrics)
        
        # Increase figure height to accommodate annotations
        fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 7))
        
        if n_clusters == 1:
            axes = [axes]
        
        vibrant_colors = ['#2196F3', '#4CAF50']  # Blue for Clinical, Green for Non-Clinical
        
        def interpret_hedges_g(g):
            """Interpret Hedge's g effect size."""
            abs_g = abs(g)
            if abs_g < 0.2:
                return "negligible"
            elif abs_g < 0.5:
                return "small"
            elif abs_g < 0.8:
                return "medium"
            else:
                return "large"
        
        for cluster_idx, cluster_id in enumerate(clusters):
            ax = axes[cluster_idx]
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster_id]
            
            x_pos = np.arange(len(metrics))
            width = 0.35
            
            clinical_means = []
            nonclinical_means = []
            clinical_stds = []
            nonclinical_stds = []
            p_values = []
            hedges_g_values = []
            
            for metric in metrics:
                metric_data = cluster_data[cluster_data['Metric'] == metric]
                if len(metric_data) > 0:
                    row = metric_data.iloc[0]
                    clinical_means.append(row['Clinical_mean'])
                    nonclinical_means.append(row['Non-Clinical_mean'])
                    clinical_stds.append(row['Clinical_std'])
                    nonclinical_stds.append(row['Non-Clinical_std'])
                    p_values.append(row.get('p_Welch', np.nan))
                    hedges_g_values.append(row.get('Hedges_g', np.nan))
                else:
                    clinical_means.append(0)
                    nonclinical_means.append(0)
                    clinical_stds.append(0)
                    nonclinical_stds.append(0)
                    p_values.append(np.nan)
                    hedges_g_values.append(np.nan)
            
            bars1 = ax.bar(x_pos - width/2, clinical_means, width, 
                          label='Clinical', color=vibrant_colors[0], alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x_pos + width/2, nonclinical_means, width,
                          label='Non-Clinical', color=vibrant_colors[1], alpha=0.8, edgecolor='black')
            
            # Add error bars
            for i, metric in enumerate(metrics):
                ax.errorbar(i - width/2, clinical_means[i], 
                           yerr=clinical_stds[i], fmt='none', color='black', capsize=3)
                ax.errorbar(i + width/2, nonclinical_means[i],
                           yerr=nonclinical_stds[i], fmt='none', color='black', capsize=3)
            
            # First, calculate maximum height needed for proper spacing
            max_height = 0
            max_annotation_height = 0
            for i in range(len(metrics)):
                # Calculate max bar height (bar + error bar)
                max_bar_height = max(clinical_means[i] + clinical_stds[i], 
                                   nonclinical_means[i] + nonclinical_stds[i])
                max_height = max(max_height, max_bar_height)
                
                # Calculate positions: rating values, then bracket, then annotations
                # Rating values go above bars with error bars
                rating_value_y = max_bar_height + 0.15
                # Bracket goes well above rating values to avoid overlap
                bracket_y = rating_value_y + 0.25
                # P-value and Hedge's g go above bracket
                p_value_y = bracket_y + 0.25
                hedges_g_y = p_value_y + 0.25
                
                # Track the highest annotation point (add padding for text height)
                max_annotation_height = max(max_annotation_height, hedges_g_y + 0.3)
            
            # Set y-axis limit before adding text to ensure nothing gets clipped
            ax.set_ylim(bottom=0, top=max_annotation_height)
            
            # Add rating values on top of bars
            for i in range(len(metrics)):
                max_bar_height = max(clinical_means[i] + clinical_stds[i], 
                                   nonclinical_means[i] + nonclinical_stds[i])
                rating_value_y = max_bar_height + 0.15
                
                # Clinical bar value
                ax.text(i - width/2, rating_value_y, 
                       f'{clinical_means[i]:.2f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
                # Non-Clinical bar value
                ax.text(i + width/2, rating_value_y, 
                       f'{nonclinical_means[i]:.2f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add p-values and Hedge's g between bars
            for i in range(len(metrics)):
                max_bar_height = max(clinical_means[i] + clinical_stds[i], 
                                   nonclinical_means[i] + nonclinical_stds[i])
                
                # Position for rating values (already added above)
                rating_value_y = max_bar_height + 0.15
                # Position for bracket (well above rating values)
                bracket_y = rating_value_y + 0.25
                
                # Position for annotations (above the bracket)
                p_value_y = bracket_y + 0.25
                hedges_g_y = p_value_y + 0.25
                
                # Format p-value
                if not np.isnan(p_values[i]):
                    if p_values[i] < 0.001:
                        p_text = "p < 0.001"
                    elif p_values[i] < 0.01:
                        p_text = f"p = {p_values[i]:.3f}"
                    else:
                        p_text = f"p = {p_values[i]:.3f}"
                else:
                    p_text = "p = N/A"
                
                # Format Hedge's g with interpretation
                if not np.isnan(hedges_g_values[i]):
                    g_interpretation = interpret_hedges_g(hedges_g_values[i])
                    g_text = f"g = {hedges_g_values[i]:.3f} ({g_interpretation})"
                else:
                    g_text = "g = N/A"
                
                # Add bracket/line connecting the bars (with vertical lines at ends)
                # Vertical lines start just above the rating values
                bracket_start_y = rating_value_y + 0.1
                ax.plot([i - width/2, i + width/2], [bracket_y, bracket_y], 
                       'k-', linewidth=1.5)
                ax.plot([i - width/2, i - width/2], [bracket_start_y, bracket_y], 
                       'k-', linewidth=1.5)
                ax.plot([i + width/2, i + width/2], [bracket_start_y, bracket_y], 
                       'k-', linewidth=1.5)
                
                # Add p-value text
                ax.text(i, p_value_y, p_text, 
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1))
                
                # Add Hedge's g text
                ax.text(i, hedges_g_y, g_text, 
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1))
            
            ax.set_xlabel('Metric', fontsize=12)
            ax.set_ylabel('Mean Rating', fontsize=12)
            if cluster_idx == 0:
                ax.set_title(f'Figure 4: Pattern Match Comparison\nCluster {cluster_id}: Clinical vs Non-Clinical', 
                           fontsize=13, fontweight='bold')
            else:
                ax.set_title(f'Cluster {cluster_id}: Clinical vs Non-Clinical', 
                           fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            demographic_var = subgroup_enhanced_results.get('demographic_var', 'subgroup')
            plt.savefig(self.figures_folder / f'subgroup_pattern_match_{demographic_var.lower()}.png',
                       dpi=self.dpi, bbox_inches=self.bbox_inches)
        
        return fig
    
    def close_all_figures(self):
        """Close all matplotlib figures."""
        plt.close('all')
