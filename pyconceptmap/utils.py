"""
Utility functions for concept mapping analysis.

This module provides utility functions for data validation, requirements checking,
and other helper functions used throughout the concept mapping analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import sys
import subprocess


def validate_data(statements: pd.DataFrame,
                 sorting_data: List[Dict],
                 ratings: pd.DataFrame,
                 demographics: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate consistency across all data files.
    
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
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'valid' boolean and 'errors'/'warnings' lists
    """
    errors = []
    warnings = []
    
    # Check statement ID consistency
    statement_ids = set(statements['StatementID'])
    
    # Check sorting data
    for sorter in sorting_data:
        for pile_name, statements_in_pile in sorter['Sorting'].items():
            for stmt_id in statements_in_pile:
                if stmt_id not in statement_ids:
                    errors.append(f"Sorter {sorter['SorterID']} references non-existent statement {stmt_id}")
    
    # Check ratings data
    rating_statement_ids = set(ratings['StatementID'])
    missing_in_ratings = statement_ids - rating_statement_ids
    if missing_in_ratings:
        warnings.append(f"Statements {missing_in_ratings} have no ratings")
    
    extra_in_ratings = rating_statement_ids - statement_ids
    if extra_in_ratings:
        errors.append(f"Ratings exist for non-existent statements {extra_in_ratings}")
    
    # Check rater consistency
    rater_ids_ratings = set(ratings['RaterID'])
    rater_ids_demographics = set(demographics['RaterID'])
    
    missing_demographics = rater_ids_ratings - rater_ids_demographics
    if missing_demographics:
        warnings.append(f"Raters {missing_demographics} have no demographics")
    
    extra_demographics = rater_ids_demographics - rater_ids_ratings
    if extra_demographics:
        warnings.append(f"Demographics exist for non-rating raters {extra_demographics}")
    
    # Check for lumpers (participants who put all statements in one pile)
    lumpers = []
    for sorter in sorting_data:
        num_piles = len(sorter['Sorting'])
        total_statements = sum(len(statements) for statements in sorter['Sorting'].values())
        
        if num_piles == 1:
            lumpers.append(f"Sorter {sorter['SorterID']} is a lumper (1 pile)")
        elif num_piles == total_statements:
            lumpers.append(f"Sorter {sorter['SorterID']} is a splitter ({num_piles} piles)")
    
    if lumpers:
        warnings.extend(lumpers)
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def check_requirements() -> Dict[str, bool]:
    """
    Check if all required packages are installed.
    
    Returns
    -------
    Dict[str, bool]
        Dictionary with package names as keys and installation status as values
    """
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy'
    ]
    
    results = {}
    
    for package in required_packages:
        try:
            __import__(package)
            results[package] = True
        except ImportError:
            results[package] = False
    
    return results


def install_requirements():
    """
    Install required packages using pip.
    """
    required_packages = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0'
    ]
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")


def create_sample_data(output_folder: Path) -> bool:
    """
    Create sample data files for testing.
    
    Parameters
    ----------
    output_folder : Path
        Path to the output folder
        
    Returns
    -------
    bool
        True if sample data created successfully
    """
    try:
        output_folder.mkdir(exist_ok=True)
        
        # Create sample statements
        statements = pd.DataFrame({
            'StatementID': range(1, 11),
            'Statement': [f'Statement {i}' for i in range(1, 11)]
        })
        statements.to_csv(output_folder / 'Statements.csv', index=False)
        
        # Create sample sorting data
        sorting_data = []
        for sorter_id in range(1, 4):
            sorting = {}
            for pile_id in range(1, 4):
                pile_name = f'Pile {pile_id}'
                statements_in_pile = list(range(pile_id, 11, 3))  # Distribute statements
                sorting[pile_name] = statements_in_pile
            sorting_data.append({
                'SorterID': sorter_id,
                'Sorting': sorting
            })
        
        # Create SortedCards.csv
        sorted_cards = []
        for sorter in sorting_data:
            for pile_name, statements_in_pile in sorter['Sorting'].items():
                row = {'SorterID': sorter['SorterID'], 'PileName': pile_name}
                for i, stmt_id in enumerate(statements_in_pile):
                    row[f'Statement_{i+1}'] = stmt_id
                sorted_cards.append(row)
        
        sorted_cards_df = pd.DataFrame(sorted_cards)
        sorted_cards_df.to_csv(output_folder / 'SortedCards.csv', index=False)
        
        # Create sample demographics
        demographics = pd.DataFrame({
            'RaterID': range(1, 4),
            'Age': [25, 30, 35],
            'Experience': ['Low', 'Medium', 'High'],
            'Department': ['A', 'B', 'A']
        })
        demographics.to_csv(output_folder / 'Demographics.csv', index=False)
        
        # Create sample ratings
        ratings = []
        for rater_id in range(1, 4):
            for stmt_id in range(1, 11):
                ratings.append({
                    'RaterID': rater_id,
                    'StatementID': stmt_id,
                    'Importance': np.random.randint(1, 6),
                    'Feasibility': np.random.randint(1, 6)
                })
        
        ratings_df = pd.DataFrame(ratings)
        ratings_df.to_csv(output_folder / 'Ratings.csv', index=False)
        
        print("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False


def calculate_silhouette_scores(mds_coords: np.ndarray, max_clusters: int = 10) -> List[float]:
    """
    Calculate silhouette scores for different numbers of clusters.
    
    Parameters
    ----------
    mds_coords : np.ndarray
        MDS coordinates
    max_clusters : int
        Maximum number of clusters to test
        
    Returns
    -------
    List[float]
        List of silhouette scores
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    
    for k in range(2, min(max_clusters + 1, len(mds_coords) // 2)):
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = clustering.fit_predict(mds_coords)
        score = silhouette_score(mds_coords, cluster_labels)
        silhouette_scores.append(score)
    
    return silhouette_scores


def calculate_elbow_scores(mds_coords: np.ndarray, max_clusters: int = 10) -> List[float]:
    """
    Calculate within-cluster sum of squares for elbow method.
    
    Parameters
    ----------
    mds_coords : np.ndarray
        MDS coordinates
    max_clusters : int
        Maximum number of clusters to test
        
    Returns
    -------
    List[float]
        List of within-cluster sum of squares
    """
    from sklearn.cluster import AgglomerativeClustering
    
    wcss_scores = []
    
    for k in range(2, min(max_clusters + 1, len(mds_coords) // 2)):
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = clustering.fit_predict(mds_coords)
        
        # Calculate within-cluster sum of squares
        wcss = 0
        for cluster_id in range(k):
            cluster_points = mds_coords[cluster_labels == cluster_id]
            if len(cluster_points) > 0:
                cluster_center = np.mean(cluster_points, axis=0)
                wcss += np.sum((cluster_points - cluster_center) ** 2)
        
        wcss_scores.append(wcss)
    
    return wcss_scores


def format_number(value: float, decimals: int = 3) -> str:
    """
    Format a number with specified decimal places.
    
    Parameters
    ----------
    value : float
        Number to format
    decimals : int
        Number of decimal places
        
    Returns
    -------
    str
        Formatted number string
    """
    return f"{value:.{decimals}f}"


def create_output_directory(base_path: Path, subfolder: str = None) -> Path:
    """
    Create output directory structure.
    
    Parameters
    ----------
    base_path : Path
        Base path for output
    subfolder : str, optional
        Subfolder name
        
    Returns
    -------
    Path
        Path to the created directory
    """
    if subfolder:
        output_dir = base_path / subfolder
    else:
        output_dir = base_path / 'output'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_progress(message: str, step: int = None, total: int = None):
    """
    Print progress message with optional step counter.
    
    Parameters
    ----------
    message : str
        Progress message
    step : int, optional
        Current step number
    total : int, optional
        Total number of steps
    """
    if step is not None and total is not None:
        print(f"[{step}/{total}] {message}")
    else:
        print(message)


def validate_file_structure(data_folder: Path) -> bool:
    """
    Validate that all required files exist in the data folder.
    
    Parameters
    ----------
    data_folder : Path
        Path to the data folder
        
    Returns
    -------
    bool
        True if all required files exist
    """
    required_files = [
        'Statements.csv',
        'SortedCards.csv',
        'Demographics.csv', 
        'Ratings.csv'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (data_folder / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True
