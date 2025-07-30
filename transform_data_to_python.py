#!/usr/bin/env python3
"""
Transform Concept Mapping Data to Python Format
===============================================

This script converts Qualtrics survey data into a format suitable for 
concept mapping analysis in Python. It handles CSV and TSV files and
extracts statements, ratings, and demographics for analysis.

This implementation is designed for researchers conducting concept mapping studies
in healthcare, education, business, or any domain requiring structured analysis
of complex ideas and their relationships.

Author: Concept Mapping Analysis Team
Date: 2025
License: Educational and Research Use
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import csv

def process_csv_file(file_path):
    """
    Process CSV file from Qualtrics survey (with two header rows).
    
    This function extracts statements, ratings, and demographics from
    a Qualtrics CSV export file with two header rows.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (statements, ratings_data, demographics_data)
    """
    print(f"Processing CSV file: {file_path}")
    
    # Read the first two rows separately
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        header1 = f.readline()
        header2 = f.readline()
    
    # Read the actual data, using the third row as header
    df = pd.read_csv(file_path, skiprows=2)
    
    # Extract statements from Q1_x columns (from header2)
    # We'll use the second header row to get statement text
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip first
        question_row = next(reader)
    
    statements = []
    for i in range(1, 101):
        col_name = f"Q1_{i}"
        if col_name in df.columns:
            # Find the statement text from the second header row
            idx = list(df.columns).index(col_name)
            statement_text = question_row[idx]
            if pd.notna(statement_text) and statement_text.strip():
                # Remove trailing :Right or similar
                statement_text = statement_text.split(':')[0].strip()
                # Remove leading numbering if present
                statement_text = statement_text.split('- ', 1)[-1].strip() if '- ' in statement_text else statement_text
                statements.append({
                    'StatementID': i,
                    'StatementText': statement_text
                })
    
    # Extract ratings data
    ratings_data = []
    demographics_data = []
    for idx, row in df.iterrows():
        participant_id = f"P{idx+1}"
        # Extract demographics (customize as needed)
        demographics_data.append({
            'ParticipantID': participant_id,
            'Age': row.get('Q101', ''),
            'Gender': row.get('Q102', ''),
            'Role': row.get('Q103', ''),
            'Experience': row.get('Q104', '')
        })
        # Extract importance and feasibility ratings
        for i in range(1, 101):
            importance_col = f"Q2.1_{i}"
            feasibility_col = f"Q2.2_{i}"
            if importance_col in df.columns and pd.notna(row[importance_col]):
                try:
                    ratings_data.append({
                        'ParticipantID': participant_id,
                        'StatementID': i,
                        'RatingType': 'Importance',
                        'Rating': int(row[importance_col])
                    })
                except (ValueError, TypeError):
                    continue
            if feasibility_col in df.columns and pd.notna(row[feasibility_col]):
                try:
                    ratings_data.append({
                        'ParticipantID': participant_id,
                        'StatementID': i,
                        'RatingType': 'Feasibility',
                        'Rating': int(row[feasibility_col])
                    })
                except (ValueError, TypeError):
                    continue
    return statements, ratings_data, demographics_data

def process_tsv_file(file_path):
    """
    Process TSV file from Qualtrics survey.
    
    This function extracts statements, ratings, and demographics from
    a Qualtrics TSV export file.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        tuple: (statements, ratings_data, demographics_data)
    """
    print(f"Processing TSV file: {file_path}")
    
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Extract statements from the first few rows
    statements = []
    for i in range(1, 101):  # 100 statements
        col_name = f"Q{i}"
        if col_name in df.columns:
            statement_text = df[col_name].iloc[0]  # First row contains statement text
            if pd.notna(statement_text) and statement_text.strip():
                statements.append({
                    'StatementID': i,
                    'StatementText': statement_text.strip()
                })
    
    # Extract ratings data
    ratings_data = []
    demographics_data = []
    
    for idx, row in df.iterrows():
        if idx < 2:  # Skip header rows
            continue
            
        participant_id = f"P{idx-1}"
        
        # Extract demographics
        demographics_data.append({
            'ParticipantID': participant_id,
            'Age': row.get('Q101', ''),
            'Gender': row.get('Q102', ''),
            'Role': row.get('Q103', ''),
            'Experience': row.get('Q104', '')
        })
        
        # Extract importance and feasibility ratings
        for i in range(1, 101):
            importance_col = f"Q{i}_1"  # Importance rating
            feasibility_col = f"Q{i}_2"  # Feasibility rating
            
            if importance_col in df.columns and feasibility_col in df.columns:
                importance = row[importance_col]
                feasibility = row[feasibility_col]
                
                if pd.notna(importance) and pd.notna(feasibility):
                    try:
                        ratings_data.append({
                            'ParticipantID': participant_id,
                            'StatementID': i,
                            'RatingType': 'Importance',
                            'Rating': int(importance)
                        })
                        ratings_data.append({
                            'ParticipantID': participant_id,
                            'StatementID': i,
                            'RatingType': 'Feasibility',
                            'Rating': int(feasibility)
                        })
                    except (ValueError, TypeError):
                        # Skip non-numeric ratings
                        continue
    
    return statements, ratings_data, demographics_data

def save_transformed_data(statements, ratings_data, demographics_data, output_dir="data/python_analysis"):
    """
    Save transformed data to CSV files.
    
    Args:
        statements (list): List of statement dictionaries
        ratings_data (list): List of rating dictionaries
        demographics_data (list): List of demographic dictionaries
        output_dir (str): Output directory for transformed data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrames
    statements_df = pd.DataFrame(statements)
    ratings_df = pd.DataFrame(ratings_data)
    demographics_df = pd.DataFrame(demographics_data)
    
    # Save to CSV files
    statements_df.to_csv(os.path.join(output_dir, "statements.csv"), index=False)
    ratings_df.to_csv(os.path.join(output_dir, "ratings.csv"), index=False)
    demographics_df.to_csv(os.path.join(output_dir, "demographics.csv"), index=False)
    
    # Create a placeholder sorted_cards file (empty for now)
    sorted_cards_df = pd.DataFrame(columns=['ParticipantID', 'StatementID', 'GroupID'])
    sorted_cards_df.to_csv(os.path.join(output_dir, "sorted_cards.csv"), index=False)
    
    print(f"✅ Transformed data saved to {output_dir}")
    print(f"   - {len(statements)} statements")
    print(f"   - {len(ratings_data)} ratings")
    print(f"   - {len(demographics_data)} participants")

def main():
    """
    Main function to transform concept mapping data.
    
    This function processes Qualtrics survey data and converts it to
    a format suitable for Python concept mapping analysis.
    """
    print("=" * 60)
    print("CONCEPT MAPPING DATA TRANSFORMATION - PYTHON")
    print("=" * 60)
    
    # Look for data files in the data directory
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        print("Please place your Qualtrics CSV/TSV files in the 'data' directory.")
        return
    
    # Find CSV and TSV files
    csv_files = list(Path(data_dir).glob("*.csv"))
    tsv_files = list(Path(data_dir).glob("*.tsv"))
    
    if not csv_files and not tsv_files:
        print(f"❌ No CSV or TSV files found in '{data_dir}'!")
        print("Please add your Qualtrics export files to the data directory.")
        return
    
    # Process the first file found
    if csv_files:
        file_path = str(csv_files[0])
        print(f"Found CSV file: {file_path}")
        statements, ratings_data, demographics_data = process_csv_file(file_path)
    elif tsv_files:
        file_path = str(tsv_files[0])
        print(f"Found TSV file: {file_path}")
        statements, ratings_data, demographics_data = process_tsv_file(file_path)
    
    # Save transformed data
    save_transformed_data(statements, ratings_data, demographics_data)
    
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION COMPLETED!")
    print("=" * 60)
    print("📁 Transformed data is ready for Python analysis")
    print("🚀 Run: python concept_mapping_analysis_python.py")

if __name__ == "__main__":
    main() 