"""
Data handling and validation for concept mapping analysis.

This module provides functionality for loading, validating, and processing
the four required CSV files for concept mapping analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings


class DataHandler:
    """
    Handles data loading and validation for concept mapping analysis.
    
    This class provides methods to load and validate the four required CSV files:
    - Statements.csv: Statement IDs and text
    - SortedCards.csv: Participant sorting data
    - Demographics.csv: Participant demographics
    - Ratings.csv: Importance and feasibility ratings
    """
    
    def __init__(self):
        """Initialize the data handler."""
        self.required_files = [
            'Statements.csv',
            'SortedCards.csv', 
            'Demographics.csv',
            'Ratings.csv'
        ]
    
    def load_statements(self, data_folder: Path) -> pd.DataFrame:
        """
        Load statements data from Statements.csv.
        
        Parameters
        ----------
        data_folder : Path
            Path to the data folder
            
        Returns
        -------
        pd.DataFrame
            Statements data with columns: StatementID, Statement
            
        Raises
        ------
        FileNotFoundError
            If Statements.csv is not found
        ValueError
            If the file format is incorrect
        """
        file_path = data_folder / 'Statements.csv'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Statements.csv not found in {data_folder}")
        
        try:
            statements = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['StatementID', 'Statement']
            missing_cols = [col for col in required_cols if col not in statements.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate StatementID is sequential from 1 to N
            expected_ids = range(1, len(statements) + 1)
            if not statements['StatementID'].equals(pd.Series(expected_ids)):
                warnings.warn("StatementID is not sequential from 1 to N")
            
            print(f"✅ Loaded {len(statements)} statements")
            return statements
            
        except Exception as e:
            raise ValueError(f"Error loading Statements.csv: {e}")
    
    def load_sorting_data(self, data_folder: Path) -> List[Dict]:
        """
        Load sorting data from SortedCards.csv.
        
        Parameters
        ----------
        data_folder : Path
            Path to the data folder
            
        Returns
        -------
        List[Dict]
            List of participant sorting data
            
        Raises
        ------
        FileNotFoundError
            If SortedCards.csv is not found
        ValueError
            If the file format is incorrect
        """
        file_path = data_folder / 'SortedCards.csv'
        
        if not file_path.exists():
            raise FileNotFoundError(f"SortedCards.csv not found in {data_folder}")
        
        try:
            sorting_df = pd.read_csv(file_path)
            
            # Validate required columns
            if 'SorterID' not in sorting_df.columns:
                raise ValueError("Missing required column: SorterID")
            
            # Process sorting data
            sorting_data = []
            for sorter_id in sorting_df['SorterID'].unique():
                sorter_data = sorting_df[sorting_df['SorterID'] == sorter_id]
                
                # Get pile names and statements
                piles = {}
                for _, row in sorter_data.iterrows():
                    pile_name = row.get('PileName', f'Pile_{len(piles)+1}')
                    statements = []
                    
                    # Extract statement IDs from the row
                    for col in sorter_data.columns:
                        if col not in ['SorterID', 'PileName'] and pd.notna(row[col]):
                            try:
                                stmt_id = int(row[col])
                                statements.append(stmt_id)
                            except (ValueError, TypeError):
                                continue
                    
                    if statements:  # Only add non-empty piles
                        piles[pile_name] = statements
                
                if piles:  # Only add sorters with data
                    sorting_data.append({
                        'SorterID': sorter_id,
                        'Sorting': piles
                    })
            
            print(f"✅ Loaded sorting data from {len(sorting_data)} sorters")
            return sorting_data
            
        except Exception as e:
            raise ValueError(f"Error loading SortedCards.csv: {e}")
    
    def load_demographics(self, data_folder: Path) -> pd.DataFrame:
        """
        Load demographics data from Demographics.csv.
        
        Parameters
        ----------
        data_folder : Path
            Path to the data folder
            
        Returns
        -------
        pd.DataFrame
            Demographics data
            
        Raises
        ------
        FileNotFoundError
            If Demographics.csv is not found
        ValueError
            If the file format is incorrect
        """
        file_path = data_folder / 'Demographics.csv'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Demographics.csv not found in {data_folder}")
        
        try:
            demographics = pd.read_csv(file_path)
            
            # Validate required columns
            if 'RaterID' not in demographics.columns:
                raise ValueError("Missing required column: RaterID")
            
            print(f"✅ Loaded demographics for {len(demographics)} participants")
            return demographics
            
        except Exception as e:
            raise ValueError(f"Error loading Demographics.csv: {e}")
    
    def load_ratings(self, data_folder: Path) -> pd.DataFrame:
        """
        Load ratings data from Ratings.csv.
        
        Parameters
        ----------
        data_folder : Path
            Path to the data folder
            
        Returns
        -------
        pd.DataFrame
            Ratings data with columns: RaterID, StatementID, and rating variables
            
        Raises
        ------
        FileNotFoundError
            If Ratings.csv is not found
        ValueError
            If the file format is incorrect
        """
        file_path = data_folder / 'Ratings.csv'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Ratings.csv not found in {data_folder}")
        
        try:
            ratings = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['RaterID', 'StatementID']
            missing_cols = [col for col in required_cols if col not in ratings.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for rating variables (should be numeric)
            rating_cols = [col for col in ratings.columns 
                         if col not in ['RaterID', 'StatementID']]
            
            if not rating_cols:
                raise ValueError("No rating variables found")
            
            # Validate rating scales (should be 1-5 or similar)
            for col in rating_cols:
                if ratings[col].dtype not in ['int64', 'float64']:
                    warnings.warn(f"Rating column {col} is not numeric")
                else:
                    min_val = ratings[col].min()
                    max_val = ratings[col].max()
                    if min_val < 1 or max_val > 5:
                        warnings.warn(f"Rating column {col} has values outside 1-5 range")
            
            print(f"✅ Loaded {len(ratings)} ratings")
            print(f"  - Rating variables: {rating_cols}")
            
            return ratings
            
        except Exception as e:
            raise ValueError(f"Error loading Ratings.csv: {e}")
    
    def validate_data_consistency(self, statements: pd.DataFrame,
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
            Dictionary with 'errors' and 'warnings' keys
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
        
        return {'errors': errors, 'warnings': warnings}
    
    def create_binary_variables(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary variables for quantitative demographic variables.
        
        Parameters
        ----------
        demographics : pd.DataFrame
            Demographics data
            
        Returns
        -------
        pd.DataFrame
            Demographics data with binary variables added
        """
        demographics_copy = demographics.copy()
        
        for col in demographics.columns:
            if col == 'RaterID':
                continue
                
            if demographics[col].dtype in ['int64', 'float64']:
                # Create binary variable
                median_val = demographics[col].median()
                binary_col = f"{col}_binary"
                demographics_copy[binary_col] = (demographics[col] > median_val).astype(int)
                print(f"  - Created binary variable: {binary_col}")
        
        return demographics_copy
    
    def check_for_lumpers(self, sorting_data: List[Dict]) -> List[Dict]:
        """
        Check for participants who put all statements in one pile (lumpers).
        
        Parameters
        ----------
        sorting_data : List[Dict]
            Sorting data
            
        Returns
        -------
        List[Dict]
            Information about lumpers
        """
        lumpers = []
        
        for sorter in sorting_data:
            num_piles = len(sorter['Sorting'])
            total_statements = sum(len(statements) for statements in sorter['Sorting'].values())
            
            if num_piles == 1:
                lumpers.append({
                    'SorterID': sorter['SorterID'],
                    'NumPiles': num_piles,
                    'TotalStatements': total_statements,
                    'Type': 'Lumper'
                })
            elif num_piles == total_statements:
                lumpers.append({
                    'SorterID': sorter['SorterID'],
                    'NumPiles': num_piles,
                    'TotalStatements': total_statements,
                    'Type': 'Splitter'
                })
        
        if lumpers:
            print(f"⚠️  Found {len(lumpers)} potential lumpers/splitters:")
            for lumper in lumpers:
                print(f"  - Sorter {lumper['SorterID']}: {lumper['Type']} "
                      f"({lumper['NumPiles']} piles, {lumper['TotalStatements']} statements)")
        
        return lumpers
