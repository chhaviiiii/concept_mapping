"""
Core concept mapping analysis functionality.

This module contains the main ConceptMappingAnalysis class that orchestrates
the entire concept mapping workflow.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

from .data_handler import DataHandler
from .visualizer import ConceptMapVisualizer
from .reporter import ReportGenerator
from .utils import validate_data, check_requirements


class ConceptMappingAnalysis:
    """
    Main class for concept mapping analysis.
    
    This class provides a comprehensive interface for performing concept mapping
    analysis, including data loading, MDS computation, clustering, visualization,
    and report generation.
    
    Parameters
    ----------
    data_folder : str or Path
        Path to the folder containing the four required CSV files
    output_folder : str or Path, optional
        Path to the output folder for results. If None, creates 'output' folder
        in the data folder
    random_state : int, optional
        Random state for reproducibility. Default is 42
        
    Attributes
    ----------
    data_handler : DataHandler
        Handles data loading and validation
    visualizer : ConceptMapVisualizer
        Handles all visualization tasks
    reporter : ReportGenerator
        Handles report generation
    statements : pd.DataFrame
        Statements data
    sorting_data : List[Dict]
        Participant sorting data
    ratings : pd.DataFrame
        Ratings data
    demographics : pd.DataFrame
        Demographics data
    mds_coords : np.ndarray
        MDS coordinates
    cluster_labels : np.ndarray
        Cluster assignments
    n_clusters : int
        Number of clusters
    """
    
    def __init__(self, data_folder: Union[str, Path], 
                 output_folder: Optional[Union[str, Path]] = None,
                 random_state: int = 42):
        """Initialize the concept mapping analysis."""
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder) if output_folder else self.data_folder / 'output'
        self.random_state = random_state
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_handler = DataHandler()
        self.visualizer = ConceptMapVisualizer(self.output_folder)
        self.reporter = ReportGenerator(self.output_folder)
        
        # Data attributes
        self.statements = None
        self.sorting_data = None
        self.ratings = None
        self.demographics = None
        self.mds_coords = None
        self.cluster_labels = None
        self.n_clusters = None
        
        # Analysis results
        self.cluster_means = None
        self.statement_summary = None
        self.anova_results = None
        self.tukey_results = None
        
        print(f"PyConceptMap initialized")
        print(f"Data folder: {self.data_folder}")
        print(f"Output folder: {self.output_folder}")
    
    def load_data(self) -> bool:
        """
        Load and validate the concept mapping data.
        
        Returns
        -------
        bool
            True if data loaded successfully, False otherwise
        """
        try:
            print("Loading concept mapping data...")
            
            # Load the four required files
            self.statements = self.data_handler.load_statements(self.data_folder)
            self.sorting_data = self.data_handler.load_sorting_data(self.data_folder)
            self.ratings = self.data_handler.load_ratings(self.data_folder)
            self.demographics = self.data_handler.load_demographics(self.data_folder)
            
            # Validate data consistency
            validation_result = validate_data(
                self.statements, self.sorting_data, 
                self.ratings, self.demographics
            )
            
            if not validation_result['valid']:
                print("Data validation failed:")
                for error in validation_result['errors']:
                    print(f"  - {error}")
                return False
            
            print("✅ Data loaded and validated successfully")
            print(f"  - {len(self.statements)} statements")
            print(f"  - {len(self.sorting_data)} sorters")
            print(f"  - {len(self.ratings)} ratings")
            print(f"  - {len(self.demographics)} participants")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def perform_mds(self, method: str = 'smacof') -> bool:
        """
        Perform multidimensional scaling on the sorting data.
        
        Parameters
        ----------
        method : str
            MDS method to use ('smacof' or 'classical')
            
        Returns
        -------
        bool
            True if MDS completed successfully
        """
        try:
            print(f"Performing MDS using {method} method...")
            
            # Create co-occurrence matrix from sorting data
            cooccurrence_matrix = self._create_cooccurrence_matrix()
            
            # Convert to distance matrix
            distance_matrix = 1.0 - cooccurrence_matrix
            np.fill_diagonal(distance_matrix, 0)
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            
            # Perform MDS
            if method == 'smacof':
                from sklearn.manifold import MDS
                mds = MDS(n_components=2, dissimilarity='precomputed', 
                         random_state=self.random_state)
                self.mds_coords = mds.fit_transform(distance_matrix)
                stress = mds.stress_
            else:
                from sklearn.manifold import MDS
                mds = MDS(n_components=2, dissimilarity='precomputed',
                         random_state=self.random_state)
                self.mds_coords = mds.fit_transform(distance_matrix)
                stress = mds.stress_
            
            print(f"✅ MDS completed successfully")
            print(f"  - Stress value: {stress:.4f}")
            print(f"  - Coordinates shape: {self.mds_coords.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error performing MDS: {e}")
            return False
    
    def perform_clustering(self, method: str = 'ward', 
                          n_clusters: Optional[int] = None,
                          auto_select: bool = True) -> bool:
        """
        Perform hierarchical clustering on MDS coordinates.
        
        Parameters
        ----------
        method : str
            Clustering method ('ward', 'complete', 'average', 'single')
        n_clusters : int, optional
            Number of clusters. If None, will be determined automatically
        auto_select : bool
            Whether to automatically select optimal number of clusters
            
        Returns
        -------
        bool
            True if clustering completed successfully
        """
        try:
            print(f"Performing clustering using {method} method...")
            
            if self.mds_coords is None:
                print("❌ MDS coordinates not found. Run perform_mds() first.")
                return False
            
            # Determine number of clusters
            if n_clusters is None and auto_select:
                n_clusters = self._select_optimal_clusters()
            elif n_clusters is None:
                n_clusters = 2  # Default
            
            # Perform clustering
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                linkage=method
            )
            self.cluster_labels = clustering.fit_predict(self.mds_coords)
            self.n_clusters = n_clusters
            
            # Add cluster information to statements
            if hasattr(self, 'statements') and self.statements is not None:
                self.statements['Cluster'] = self.cluster_labels + 1  # 1-indexed
            
            print(f"✅ Clustering completed successfully")
            print(f"  - Number of clusters: {self.n_clusters}")
            print(f"  - Cluster sizes: {np.bincount(self.cluster_labels)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error performing clustering: {e}")
            return False
    
    def analyze_ratings(self) -> bool:
        """
        Analyze importance and feasibility ratings.
        
        Returns
        -------
        bool
            True if analysis completed successfully
        """
        try:
            print("Analyzing ratings...")
            
            if self.ratings is None:
                print("❌ Ratings data not found.")
                return False
            
            # Calculate statement-level statistics
            self.statement_summary = self._calculate_statement_statistics()
            
            # Calculate cluster-level statistics
            self.cluster_means = self._calculate_cluster_statistics()
            
            # Perform ANOVA
            self.anova_results = self._perform_anova()
            
            # Perform Tukey's HSD test
            self.tukey_results = self._perform_tukey_hsd()
            
            print("✅ Rating analysis completed successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing ratings: {e}")
            return False
    
    def generate_visualizations(self, save_plots: bool = True) -> bool:
        """
        Generate all concept mapping visualizations.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save plots to files
            
        Returns
        -------
        bool
            True if visualizations generated successfully
        """
        try:
            print("Generating visualizations...")
            
            if self.mds_coords is None or self.cluster_labels is None:
                print("❌ MDS or clustering not completed. Run analysis first.")
                return False
            
            # Generate core visualizations
            self.visualizer.create_point_map(
                self.mds_coords, self.statements, save=save_plots
            )
            
            self.visualizer.create_cluster_map(
                self.mds_coords, self.cluster_labels, self.statements, save=save_plots
            )
            
            self.visualizer.create_point_rating_map(
                self.mds_coords, self.statement_summary, save=save_plots
            )
            
            self.visualizer.create_cluster_rating_map(
                self.cluster_means, save=save_plots
            )
            
            self.visualizer.create_pattern_match(
                self.cluster_means, save=save_plots
            )
            
            self.visualizer.create_go_zone_plot(
                self.statement_summary, save=save_plots
            )
            
            # Generate additional visualizations
            self.visualizer.create_dendrogram(
                self.mds_coords, self.cluster_labels, save=save_plots
            )
            
            self.visualizer.create_parallel_coordinates(
                self.cluster_means, save=save_plots
            )
            
            print("✅ Visualizations generated successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            return False
    
    def generate_reports(self) -> bool:
        """
        Generate comprehensive reports.
        
        Returns
        -------
        bool
            True if reports generated successfully
        """
        try:
            print("Generating reports...")
            
            # Generate summary reports
            self.reporter.generate_sorter_summary(self.sorting_data)
            self.reporter.generate_rater_summary(self.demographics)
            self.reporter.generate_statement_summary(self.statement_summary, self.n_clusters)
            
            # Generate analysis reports
            if self.anova_results is not None:
                self.reporter.generate_anova_report(self.anova_results)
            
            if self.tukey_results is not None:
                self.reporter.generate_tukey_report(self.tukey_results)
            
            print("✅ Reports generated successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating reports: {e}")
            return False
    
    def run_complete_analysis(self, mds_method: str = 'smacof',
                            clustering_method: str = 'ward',
                            auto_select_clusters: bool = True) -> bool:
        """
        Run the complete concept mapping analysis workflow.
        
        Parameters
        ----------
        mds_method : str
            MDS method to use
        clustering_method : str
            Clustering method to use
        auto_select_clusters : bool
            Whether to automatically select optimal number of clusters
            
        Returns
        -------
        bool
            True if complete analysis successful
        """
        print("=" * 60)
        print("PYCONCEPTMAP: COMPLETE ANALYSIS WORKFLOW")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Perform MDS
        if not self.perform_mds(method=mds_method):
            return False
        
        # Step 3: Perform clustering
        if not self.perform_clustering(method=clustering_method, 
                                     auto_select=auto_select_clusters):
            return False
        
        # Step 4: Analyze ratings
        if not self.analyze_ratings():
            return False
        
        # Step 5: Generate visualizations
        if not self.generate_visualizations():
            return False
        
        # Step 6: Generate reports
        if not self.generate_reports():
            return False
        
        print("=" * 60)
        print("✅ COMPLETE ANALYSIS FINISHED SUCCESSFULLY")
        print("=" * 60)
        
        return True
    
    def _create_cooccurrence_matrix(self) -> np.ndarray:
        """Create co-occurrence matrix from sorting data."""
        n_statements = len(self.statements)
        cooccurrence_matrix = np.zeros((n_statements, n_statements))
        
        for participant in self.sorting_data:
            sorting = participant['Sorting']
            
            # For each pile, increment co-occurrence for all pairs within that pile
            for pile_name, statements_in_pile in sorting.items():
                # statements_in_pile is already a list of statement IDs
                for i, stmt1 in enumerate(statements_in_pile):
                    for j, stmt2 in enumerate(statements_in_pile):
                        if stmt1 <= n_statements and stmt2 <= n_statements:
                            idx1 = stmt1 - 1
                            idx2 = stmt2 - 1
                            cooccurrence_matrix[idx1, idx2] += 1
        
        # Normalize by number of participants
        n_participants = len(self.sorting_data)
        if n_participants > 0:
            cooccurrence_matrix = cooccurrence_matrix / n_participants
        
        return cooccurrence_matrix
    
    def _select_optimal_clusters(self) -> int:
        """Select optimal number of clusters using silhouette analysis."""
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import AgglomerativeClustering
        
        silhouette_scores = []
        max_clusters = min(10, len(self.statements) // 2)
        
        for k in range(2, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
            cluster_labels = clustering.fit_predict(self.mds_coords)
            silhouette_scores.append(silhouette_score(self.mds_coords, cluster_labels))
        
        optimal_k = np.argmax(silhouette_scores) + 2
        print(f"  - Optimal clusters (silhouette): {optimal_k}")
        
        return optimal_k
    
    def _calculate_statement_statistics(self) -> pd.DataFrame:
        """Calculate statement-level statistics."""
        if self.ratings is None:
            return pd.DataFrame()
        
        # Get rating columns (exclude RaterID and StatementID)
        rating_cols = [col for col in self.ratings.columns 
                      if col not in ['RaterID', 'StatementID']]
        
        # Calculate statistics for each statement
        statement_stats = []
        for stmt_id in self.statements['StatementID']:
            stmt_ratings = self.ratings[self.ratings['StatementID'] == stmt_id]
            
            row = {'StatementID': stmt_id}
            for col in rating_cols:
                if col in stmt_ratings.columns:
                    ratings_data = stmt_ratings[col].dropna()
                    if len(ratings_data) > 0:
                        row[f'{col}_mean'] = ratings_data.mean()
                        row[f'{col}_std'] = ratings_data.std()
                        row[f'{col}_count'] = len(ratings_data)
                        row[f'{col}_min'] = ratings_data.min()
                        row[f'{col}_max'] = ratings_data.max()
                    else:
                        row[f'{col}_mean'] = np.nan
                        row[f'{col}_std'] = np.nan
                        row[f'{col}_count'] = 0
                        row[f'{col}_min'] = np.nan
                        row[f'{col}_max'] = np.nan
            
            statement_stats.append(row)
        
        return pd.DataFrame(statement_stats)
    
    def _calculate_cluster_statistics(self) -> pd.DataFrame:
        """Calculate cluster-level statistics."""
        if self.statement_summary is None or self.cluster_labels is None:
            return pd.DataFrame()
        
        # Add cluster information to statement summary
        statement_summary_with_clusters = self.statement_summary.copy()
        statement_summary_with_clusters['Cluster'] = self.cluster_labels + 1
        
        # Get rating columns
        rating_cols = [col for col in statement_summary_with_clusters.columns 
                      if col.endswith('_mean') and not col.startswith('StatementID')]
        
        cluster_stats = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = statement_summary_with_clusters['Cluster'] == (cluster_id + 1)
            cluster_data = statement_summary_with_clusters[cluster_mask]
            
            row = {'Cluster': cluster_id + 1}
            for col in rating_cols:
                if col in cluster_data.columns:
                    values = cluster_data[col].dropna()
                    if len(values) > 0:
                        row[col.replace('_mean', '')] = values.mean()
                    else:
                        row[col.replace('_mean', '')] = np.nan
            
            cluster_stats.append(row)
        
        return pd.DataFrame(cluster_stats)
    
    def _perform_anova(self) -> Dict:
        """Perform ANOVA analysis."""
        if self.ratings is None or self.cluster_labels is None:
            return {}
        
        anova_results = {}
        rating_cols = [col for col in self.ratings.columns 
                      if col not in ['RaterID', 'StatementID']]
        
        for col in rating_cols:
            if col in self.ratings.columns:
                # Prepare data for ANOVA
                anova_data = []
                for cluster_id in range(self.n_clusters):
                    cluster_statements = np.where(self.cluster_labels == cluster_id)[0] + 1
                    cluster_ratings = self.ratings[
                        (self.ratings['StatementID'].isin(cluster_statements)) & 
                        (self.ratings[col].notna())
                    ][col].values
                    anova_data.append(cluster_ratings)
                
                # Perform ANOVA
                try:
                    from scipy import stats
                    f_statistic, p_value = stats.f_oneway(*anova_data)
                    
                    # Calculate degrees of freedom and sum of squares
                    n_total = sum(len(group) for group in anova_data)
                    n_groups = len(anova_data)
                    
                    # Calculate means
                    grand_mean = np.mean([np.mean(group) for group in anova_data if len(group) > 0])
                    
                    # Between-group sum of squares
                    ss_between = 0
                    for group in anova_data:
                        if len(group) > 0:
                            group_mean = np.mean(group)
                            ss_between += len(group) * (group_mean - grand_mean) ** 2
                    
                    # Within-group sum of squares
                    ss_within = 0
                    for group in anova_data:
                        if len(group) > 0:
                            group_mean = np.mean(group)
                            ss_within += np.sum((group - group_mean) ** 2)
                    
                    anova_results[col] = {
                        'f_statistic': f_statistic,
                        'p_value': p_value,
                        'df_between': n_groups - 1,
                        'df_within': n_total - n_groups,
                        'ss_between': ss_between,
                        'ss_within': ss_within,
                        'ms_between': ss_between / (n_groups - 1) if n_groups > 1 else 0,
                        'ms_within': ss_within / (n_total - n_groups) if n_total > n_groups else 0
                    }
                except Exception as e:
                    print(f"Warning: Could not perform ANOVA for {col}: {e}")
                    anova_results[col] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'df_between': 0,
                        'df_within': 0,
                        'ss_between': 0,
                        'ss_within': 0,
                        'ms_between': 0,
                        'ms_within': 0
                    }
        
        return anova_results
    
    def _perform_tukey_hsd(self) -> Dict:
        """Perform Tukey's HSD test."""
        if self.ratings is None or self.cluster_labels is None:
            return {}
        
        tukey_results = {}
        rating_cols = [col for col in self.ratings.columns 
                      if col not in ['RaterID', 'StatementID']]
        
        for col in rating_cols:
            if col in self.ratings.columns:
                try:
                    # Prepare data for Tukey's HSD
                    tukey_data = []
                    tukey_labels = []
                    
                    for cluster_id in range(self.n_clusters):
                        cluster_statements = np.where(self.cluster_labels == cluster_id)[0] + 1
                        cluster_ratings = self.ratings[
                            (self.ratings['StatementID'].isin(cluster_statements)) & 
                            (self.ratings[col].notna())
                        ][col].values
                        
                        if len(cluster_ratings) > 0:
                            tukey_data.extend(cluster_ratings)
                            tukey_labels.extend([f'Cluster_{cluster_id+1}'] * len(cluster_ratings))
                    
                    if len(tukey_data) > 0:
                        # Perform Tukey's HSD
                        try:
                            from scipy.stats import tukey_hsd
                            cluster_groups = []
                            for i in range(self.n_clusters):
                                cluster_statements = np.where(self.cluster_labels == i)[0] + 1
                                cluster_ratings = self.ratings[
                                    (self.ratings['StatementID'].isin(cluster_statements)) & 
                                    (self.ratings[col].notna())
                                ][col].values
                                if len(cluster_ratings) > 0:
                                    cluster_groups.append(cluster_ratings)
                            
                            if len(cluster_groups) >= 2:
                                result = tukey_hsd(*cluster_groups)
                                
                                # Format results
                                comparisons = []
                                cluster_names = [f'Cluster_{i+1}' for i in range(len(cluster_groups))]
                                
                                for i in range(len(cluster_names)):
                                    for j in range(i+1, len(cluster_names)):
                                        comparisons.append({
                                            'pair': (cluster_names[i], cluster_names[j]),
                                            'difference': result.statistic[i, j],
                                            'lower_bound': result.confidence_interval.low[i, j],
                                            'upper_bound': result.confidence_interval.high[i, j],
                                            'p_adjusted': result.pvalue[i, j]
                                        })
                                
                                tukey_results[col] = {
                                    'comparisons': comparisons
                                }
                            else:
                                tukey_results[col] = {'comparisons': []}
                        except Exception as tukey_error:
                            print(f"Warning: Tukey's HSD failed for {col}: {tukey_error}")
                            tukey_results[col] = {'comparisons': []}
                    else:
                        tukey_results[col] = {'comparisons': []}
                        
                except Exception as e:
                    print(f"Warning: Could not perform Tukey's HSD for {col}: {e}")
                    tukey_results[col] = {'comparisons': []}
        
        return tukey_results
