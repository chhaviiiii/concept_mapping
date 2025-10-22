"""
Visualization functionality for concept mapping analysis.

This module provides comprehensive visualization capabilities for concept mapping,
including point maps, cluster maps, rating maps, and various statistical plots.
"""

import matplotlib.pyplot as plt
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
        
        # Plot points
        ax.scatter(mds_coords[:, 0], mds_coords[:, 1], 
                  c='steelblue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add statement numbers
        for i, (x, y) in enumerate(mds_coords):
            ax.annotate(str(i + 1), (x, y), xytext=(2, 2), 
                       textcoords='offset points', fontsize=8, ha='left')
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('Point Map (MDS Configuration)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save:
            plt.savefig(self.output_folder / 'point_map.png', 
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
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
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
        ax.set_title('Cluster Map', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save:
            plt.savefig(self.output_folder / 'cluster_map.png',
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
            # Normalize ratings for point sizes
            ratings = statement_summary[rating_col]
            min_rating = ratings.min()
            max_rating = ratings.max()
            if max_rating > min_rating:
                normalized_ratings = (ratings - min_rating) / (max_rating - min_rating)
                point_sizes = 50 + normalized_ratings * 200  # 50-250 range
            else:
                point_sizes = 100  # Default size
        else:
            point_sizes = 100  # Default size
        
        # Plot points
        scatter = ax.scatter(mds_coords[:, 0], mds_coords[:, 1],
                           s=point_sizes, c='steelblue', alpha=0.7,
                           edgecolors='black', linewidth=0.5)
        
        # Add statement numbers
        for i, (x, y) in enumerate(mds_coords):
            ax.annotate(str(i + 1), (x, y), xytext=(2, 2),
                       textcoords='offset points', fontsize=8, ha='left')
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'Point Rating Map ({rating_var})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save:
            plt.savefig(self.output_folder / f'point_rating_map_{rating_var.lower()}.png',
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
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
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
        ax.set_title('Cluster Rating Map', fontsize=16, fontweight='bold')
        
        if save:
            plt.savefig(self.output_folder / 'cluster_rating_map.png',
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
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_means)))
        
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
        ax.set_title('Pattern Match Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Rating')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_folder / 'pattern_match.png',
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
        
        # Create scatter plot
        scatter = ax.scatter(statement_summary[feas_col], 
                           statement_summary[imp_col],
                           c='steelblue', s=50, alpha=0.7,
                           edgecolors='black', linewidth=0.5)
        
        # Add quadrant lines
        ax.axhline(y=imp_mean, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=feas_mean, color='red', linestyle='--', alpha=0.7)
        
        # Add quadrant labels
        ax.text(0.02, 0.98, 'High Importance\nLow Feasibility', 
               transform=ax.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.text(0.98, 0.98, 'High Importance\nHigh Feasibility', 
               transform=ax.transAxes, fontsize=10, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        ax.text(0.02, 0.02, 'Low Importance\nLow Feasibility', 
               transform=ax.transAxes, fontsize=10, va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        ax.text(0.98, 0.02, 'Low Importance\nHigh Feasibility', 
               transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        ax.set_xlabel('Feasibility', fontsize=12)
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Go-Zone Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_folder / 'go_zone_plot.png',
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
            plt.savefig(self.output_folder / 'dendrogram.png',
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
        
        # Plot parallel coordinates
        for i, (cluster_id, row) in enumerate(normalized_data.iterrows()):
            ax.plot(range(len(numeric_cols)), row.values, 
                   marker='o', linewidth=2, label=f'Cluster {cluster_id+1}')
        
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45)
        ax.set_ylabel('Normalized Rating', fontsize=12)
        ax.set_title('Parallel Coordinates Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_folder / 'parallel_coordinates.png',
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
    
    def close_all_figures(self):
        """Close all matplotlib figures."""
        plt.close('all')
