"""
Report generation functionality for concept mapping analysis.

This module provides comprehensive report generation capabilities including
summary statistics, ANOVA results, and detailed analysis reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.metrics import silhouette_score
import warnings


class ReportGenerator:
    """
    Generates comprehensive reports for concept mapping analysis.
    
    This class provides methods to create various types of reports including:
    - Sorter summaries
    - Rater summaries  
    - Statement summaries
    - ANOVA results
    - Tukey's HSD results
    - Cluster analysis reports
    """
    
    def __init__(self, output_folder: Path):
        """
        Initialize the report generator.
        
        Parameters
        ----------
        output_folder : Path
            Path to the output folder for saving reports
        """
        self.output_folder = output_folder
        self.output_folder.mkdir(exist_ok=True)
    
    def generate_sorter_summary(self, sorting_data: List[Dict]) -> str:
        """
        Generate a summary report for sorters.
        
        Parameters
        ----------
        sorting_data : List[Dict]
            Sorting data from participants
            
        Returns
        -------
        str
            Summary report text
        """
        report_lines = ["SORTER SUMMARY REPORT", "=" * 50, ""]
        
        for sorter in sorting_data:
            sorter_id = sorter['SorterID']
            sorting = sorter['Sorting']
            
            total_statements = sum(len(statements) for statements in sorting.values())
            num_piles = len(sorting)
            
            report_lines.append(f"Sorter {sorter_id} sorted {total_statements} cards into {num_piles} piles")
        
        report_lines.extend(["", "Press any key to continue."])
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open(self.output_folder / 'sorter_summary.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Sorter summary report generated")
        return report_text
    
    def generate_rater_summary(self, demographics: pd.DataFrame) -> str:
        """
        Generate a summary report for raters.
        
        Parameters
        ----------
        demographics : pd.DataFrame
            Demographics data
            
        Returns
        -------
        str
            Summary report text
        """
        report_lines = ["RATER SUMMARY REPORT", "=" * 50, ""]
        
        for col in demographics.columns:
            if col == 'RaterID':
                continue
                
            report_lines.append(f"{col:15s}")
            
            if demographics[col].dtype in ['int64', 'float64']:
                # Numeric variable - show five number summary
                stats_summary = demographics[col].describe()
                report_lines.append(f"Min.   :{stats_summary['min']:8.3f}")
                report_lines.append(f"1st Qu.:{stats_summary['25%']:8.3f}")
                report_lines.append(f"Median :{stats_summary['50%']:8.3f}")
                report_lines.append(f"Mean   :{stats_summary['mean']:8.3f}")
                report_lines.append(f"3rd Qu.:{stats_summary['75%']:8.3f}")
                report_lines.append(f"Max.   :{stats_summary['max']:8.3f}")
            else:
                # Categorical variable - show counts
                value_counts = demographics[col].value_counts()
                for value, count in value_counts.items():
                    report_lines.append(f"{str(value):15s}:{count:3d}")
            
            report_lines.append("")
        
        report_lines.append("Press any key to continue.")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open(self.output_folder / 'rater_summary.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Rater summary report generated")
        return report_text
    
    def generate_statement_summary(self, statement_summary: pd.DataFrame,
                                 n_clusters: int) -> str:
        """
        Generate a summary report for statements.
        
        Parameters
        ----------
        statement_summary : pd.DataFrame
            Statement summary statistics
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        str
            Summary report text
        """
        # Save to CSV
        output_file = self.output_folder / f'StatementSummary{n_clusters:02d}.csv'
        statement_summary.to_csv(output_file, index=False)
        
        print(f"✅ Statement summary saved to {output_file}")
        
        # Generate text report
        report_lines = ["STATEMENT SUMMARY REPORT", "=" * 50, ""]
        report_lines.append(f"Number of clusters: {n_clusters}")
        report_lines.append(f"Total statements: {len(statement_summary)}")
        report_lines.append("")
        
        # Cluster summaries
        if 'Cluster' in statement_summary.columns:
            cluster_counts = statement_summary['Cluster'].value_counts().sort_index()
            report_lines.append("Cluster sizes:")
            for cluster_id, count in cluster_counts.items():
                report_lines.append(f"  Cluster {cluster_id}: {count} statements")
            report_lines.append("")
        
        # Rating summaries
        rating_cols = [col for col in statement_summary.columns 
                       if col not in ['StatementID', 'Statement', 'Cluster']]
        
        if rating_cols:
            report_lines.append("Rating summaries:")
            for col in rating_cols:
                if col in statement_summary.columns:
                    mean_val = statement_summary[col].mean()
                    std_val = statement_summary[col].std()
                    report_lines.append(f"  {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save text report
        with open(self.output_folder / f'statement_summary_{n_clusters:02d}.txt', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def generate_anova_report(self, anova_results: Dict) -> str:
        """
        Generate ANOVA results report.
        
        Parameters
        ----------
        anova_results : Dict
            ANOVA results dictionary
            
        Returns
        -------
        str
            ANOVA report text
        """
        report_lines = ["ANOVA RESULTS REPORT", "=" * 50, ""]
        
        for rating_var, results in anova_results.items():
            report_lines.append(f"Analysis of Variance: Response = {rating_var}")
            report_lines.append(f"{'':>10s} {'Df':>8s} {'Sum Sq':>10s} {'Mean Sq':>10s} {'F value':>10s} {'Pr(>F)':>10s}")
            report_lines.append("-" * 60)
            
            # Cluster row
            report_lines.append(f"{'Cluster':>10s} {results['df_between']:>8d} "
                              f"{results['ss_between']:>10.1f} {results['ms_between']:>10.3f} "
                              f"{results['f_statistic']:>10.3f} {results['p_value']:>10.3f}")
            
            # Residuals row
            report_lines.append(f"{'Residuals':>10s} {results['df_within']:>8d} "
                              f"{results['ss_within']:>10.1f} {results['ms_within']:>10.3f}")
            
            # Significance codes
            if results['p_value'] < 0.001:
                sig_code = "***"
            elif results['p_value'] < 0.01:
                sig_code = "**"
            elif results['p_value'] < 0.05:
                sig_code = "*"
            elif results['p_value'] < 0.1:
                sig_code = "."
            else:
                sig_code = ""
            
            report_lines.append(f"{'':>10s} {'':>8s} {'':>10s} {'':>10s} {'':>10s} {sig_code:>10s}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open(self.output_folder / 'anova_results.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ ANOVA results report generated")
        return report_text
    
    def generate_tukey_report(self, tukey_results: Dict) -> str:
        """
        Generate Tukey's HSD results report.
        
        Parameters
        ----------
        tukey_results : Dict
            Tukey's HSD results dictionary
            
        Returns
        -------
        str
            Tukey's HSD report text
        """
        report_lines = ["TUKEY'S HSD RESULTS REPORT", "=" * 50, ""]
        
        for rating_var, results in tukey_results.items():
            report_lines.append(f"Tukey multiple comparisons of means")
            report_lines.append(f"95% family-wise confidence level")
            report_lines.append("")
            report_lines.append(f"Analysis of Variance: Response = {rating_var}")
            report_lines.append("")
            
            # Format comparison results
            if 'comparisons' in results:
                report_lines.append("$Cluster")
                for comparison in results['comparisons']:
                    cluster1, cluster2 = comparison['pair']
                    diff = comparison['difference']
                    lwr = comparison['lower_bound']
                    upr = comparison['upper_bound']
                    p_adj = comparison['p_adjusted']
                    
                    report_lines.append(f"{cluster1}-{cluster2:>3s} {diff:>10.6f} "
                                      f"{lwr:>10.6f} {upr:>10.6f} {p_adj:>10.6f}")
            
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open(self.output_folder / 'tukey_results.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Tukey's HSD results report generated")
        return report_text
    
    def generate_cluster_analysis_report(self, mds_coords: np.ndarray,
                                       cluster_labels: np.ndarray,
                                       n_clusters: int) -> str:
        """
        Generate cluster analysis report.
        
        Parameters
        ----------
        mds_coords : np.ndarray
            MDS coordinates
        cluster_labels : np.ndarray
            Cluster assignments
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        str
            Cluster analysis report text
        """
        report_lines = ["CLUSTER ANALYSIS REPORT", "=" * 50, ""]
        
        # Basic statistics
        report_lines.append(f"Number of clusters: {n_clusters}")
        report_lines.append(f"Total statements: {len(mds_coords)}")
        report_lines.append("")
        
        # Cluster sizes
        cluster_counts = np.bincount(cluster_labels)
        report_lines.append("Cluster sizes:")
        for i, count in enumerate(cluster_counts):
            percentage = (count / len(cluster_labels)) * 100
            report_lines.append(f"  Cluster {i+1}: {count} statements ({percentage:.1f}%)")
        report_lines.append("")
        
        # Silhouette score
        try:
            silhouette_avg = silhouette_score(mds_coords, cluster_labels)
            report_lines.append(f"Average silhouette score: {silhouette_avg:.3f}")
        except:
            report_lines.append("Silhouette score: Could not be calculated")
        report_lines.append("")
        
        # Cluster centers
        report_lines.append("Cluster centers (MDS coordinates):")
        for i in range(n_clusters):
            mask = cluster_labels == i
            center = np.mean(mds_coords[mask], axis=0)
            report_lines.append(f"  Cluster {i+1}: ({center[0]:.3f}, {center[1]:.3f})")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open(self.output_folder / 'cluster_analysis.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Cluster analysis report generated")
        return report_text
    
    def generate_comprehensive_report(self, statements: pd.DataFrame,
                                    sorting_data: List[Dict],
                                    ratings: pd.DataFrame,
                                    demographics: pd.DataFrame,
                                    mds_coords: np.ndarray,
                                    cluster_labels: np.ndarray,
                                    statement_summary: pd.DataFrame,
                                    anova_results: Optional[Dict] = None,
                                    tukey_results: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive report combining all analyses.
        
        Parameters
        ----------
        statements : pd.DataFrame
            Statements data
        sorting_data : List[Dict]
            Sorting data
        ratings : pd.DataFrame
            Ratings data
        demographics : pd.DataFrame
            Demographics data
        mds_coords : np.ndarray
            MDS coordinates
        cluster_labels : np.ndarray
            Cluster assignments
        statement_summary : pd.DataFrame
            Statement summary statistics
        anova_results : Dict, optional
            ANOVA results
        tukey_results : Dict, optional
            Tukey's HSD results
            
        Returns
        -------
        str
            Comprehensive report text
        """
        report_lines = [
            "PYCONCEPTMAP COMPREHENSIVE ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"PyConceptMap Version: 0.1.0",
            "",
            "DATA SUMMARY",
            "-" * 20,
            f"Number of statements: {len(statements)}",
            f"Number of sorters: {len(sorting_data)}",
            f"Number of raters: {len(demographics)}",
            f"Number of ratings: {len(ratings)}",
            "",
            "CLUSTER ANALYSIS",
            "-" * 20,
            f"Number of clusters: {len(np.unique(cluster_labels))}",
            f"Cluster sizes: {np.bincount(cluster_labels).tolist()}",
            ""
        ]
        
        # Add ANOVA results if available
        if anova_results:
            report_lines.extend(["ANOVA RESULTS", "-" * 20])
            for rating_var, results in anova_results.items():
                report_lines.append(f"{rating_var}: F={results['f_statistic']:.3f}, "
                                  f"p={results['p_value']:.3f}")
            report_lines.append("")
        
        # Add Tukey results if available
        if tukey_results:
            report_lines.extend(["TUKEY'S HSD RESULTS", "-" * 20])
            for rating_var, results in tukey_results.items():
                report_lines.append(f"{rating_var}: {len(results.get('comparisons', []))} "
                                  f"pairwise comparisons")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open(self.output_folder / 'comprehensive_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Comprehensive report generated")
        return report_text
    
    def generate_subgroup_report(self, subgroup_results: Dict) -> str:
        """
        Generate subgroup analysis report.
        
        Parameters
        ----------
        subgroup_results : Dict
            Subgroup analysis results dictionary
            
        Returns
        -------
        str
            Subgroup analysis report text
        """
        if not subgroup_results or 'results' not in subgroup_results:
            return ""
        
        demographic_var = subgroup_results.get('demographic_var', 'Subgroup')
        results = subgroup_results['results']
        
        report_lines = [
            "SUBGROUP ANALYSIS REPORT",
            "=" * 60,
            f"Demographic Variable: {demographic_var}",
            ""
        ]
        
        for cluster_key in sorted(results.keys()):
            cluster_id = cluster_key.split('_')[1]
            cluster_data = results[cluster_key]
            
            report_lines.append(f"CLUSTER {cluster_id}")
            report_lines.append("-" * 60)
            
            for rating_var in sorted(cluster_data.keys()):
                var_data = cluster_data[rating_var]
                stats = var_data['subgroup_stats']
                test_results = var_data['test_results']
                subgroups = var_data['subgroups']
                
                report_lines.append(f"\n{rating_var} Ratings:")
                report_lines.append(f"{'Subgroup':<20s} {'Mean':>10s} {'Std':>10s} {'Count':>10s} {'Min':>8s} {'Max':>8s}")
                report_lines.append("-" * 66)
                
                for subgroup in subgroups:
                    s = stats[subgroup]
                    report_lines.append(
                        f"{str(subgroup):<20s} {s['mean']:>10.3f} {s['std']:>10.3f} "
                        f"{s['count']:>10d} {s['min']:>8.1f} {s['max']:>8.1f}"
                    )
                
                # Add statistical test results
                if test_results:
                    report_lines.append("")
                    if test_results['test_type'] == 't-test':
                        report_lines.append(f"Statistical Test: Independent t-test")
                        report_lines.append(f"  t-statistic: {test_results['statistic']:.3f}")
                        report_lines.append(f"  p-value: {test_results['p_value']:.4f}")
                        report_lines.append(f"  Group 1 ({subgroups[0]}): Mean = {test_results['group1_mean']:.3f}, n = {test_results['group1_n']}")
                        report_lines.append(f"  Group 2 ({subgroups[1]}): Mean = {test_results['group2_mean']:.3f}, n = {test_results['group2_n']}")
                    elif test_results['test_type'] == 'ANOVA':
                        report_lines.append(f"Statistical Test: One-way ANOVA")
                        report_lines.append(f"  F-statistic: {test_results['statistic']:.3f}")
                        report_lines.append(f"  p-value: {test_results['p_value']:.4f}")
                        report_lines.append(f"  Number of groups: {test_results['n_groups']}")
                    
                    # Significance interpretation
                    p_val = test_results['p_value']
                    if p_val < 0.001:
                        sig_text = "*** (p < 0.001) - Highly significant"
                    elif p_val < 0.01:
                        sig_text = "** (p < 0.01) - Very significant"
                    elif p_val < 0.05:
                        sig_text = "* (p < 0.05) - Significant"
                    else:
                        sig_text = "ns (p >= 0.05) - Not significant"
                    
                    report_lines.append(f"  Significance: {sig_text}")
                
                report_lines.append("")
            
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        filename = f'subgroup_analysis_{demographic_var.lower()}.txt'
        with open(self.output_folder / filename, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Subgroup analysis report generated: {filename}")
        return report_text