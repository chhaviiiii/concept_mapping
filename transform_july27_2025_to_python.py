#!/usr/bin/env python3
"""
Transform July 27, 2025 BCCS AI Workshop Data to Python Format
This script converts Qualtrics survey data into a format suitable for concept mapping analysis in Python.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

def process_csv_file(file_path):
    """Process CSV file from Qualtrics survey"""
    print(f"Processing CSV file: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
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
    
    return statements, ratings_data, demographics_data

def process_tsv_file(file_path):
    """Process TSV file from Qualtrics survey"""
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
    
    return statements, ratings_data, demographics_data

def main():
    """Main transformation function"""
    print("=== BCCS AI Workshop July 27, 2025 Data Transformation ===")
    
    # Input files
    input_files = [
        "data/BCCS AI Workshop_July 27, 2025_15.23.csv",
        "data/BCCS AI Workshop_July 27, 2025_15.26_utf8.tsv"
    ]
    
    # Output directory
    output_dir = "data/python_july27_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all data
    all_statements = []
    all_ratings = []
    all_demographics = []
    
    # Process each input file
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        if file_path.endswith('.csv'):
            statements, ratings, demographics = process_csv_file(file_path)
        elif file_path.endswith('.tsv'):
            statements, ratings, demographics = process_tsv_file(file_path)
        else:
            print(f"Warning: Unsupported file format: {file_path}")
            continue
        
        all_statements.extend(statements)
        all_ratings.extend(ratings)
        all_demographics.extend(demographics)
    
    # Remove duplicates and create final datasets
    statements_df = pd.DataFrame(all_statements).drop_duplicates(subset=['StatementID']).sort_values('StatementID')
    ratings_df = pd.DataFrame(all_ratings)
    demographics_df = pd.DataFrame(all_demographics).drop_duplicates(subset=['ParticipantID'])
    
    # Create sorted cards data (simplified - you may need to adjust based on your grouping data)
    sorted_cards_data = []
    for _, row in ratings_df.iterrows():
        if row['RatingType'] == 'Importance' and row['Rating'] >= 4:  # High importance items
            sorted_cards_data.append({
                'ParticipantID': row['ParticipantID'],
                'PileID': 1,  # Default pile for high importance
                'StatementID': row['StatementID']
            })
    
    sorted_cards_df = pd.DataFrame(sorted_cards_data)
    
    # Save the transformed data
    statements_df.to_csv(os.path.join(output_dir, "statements.csv"), index=False)
    ratings_df.to_csv(os.path.join(output_dir, "ratings.csv"), index=False)
    demographics_df.to_csv(os.path.join(output_dir, "demographics.csv"), index=False)
    sorted_cards_df.to_csv(os.path.join(output_dir, "sorted_cards.csv"), index=False)
    
    # Print summary
    print(f"\n=== Transformation Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Statements: {len(statements_df)}")
    print(f"Ratings: {len(ratings_df)}")
    print(f"Participants: {len(demographics_df)}")
    print(f"Sorted cards: {len(sorted_cards_df)}")
    
    # Create summary statistics
    summary_stats = {
        'total_statements': len(statements_df),
        'total_participants': len(demographics_df),
        'total_ratings': len(ratings_df),
        'importance_ratings': len(ratings_df[ratings_df['RatingType'] == 'Importance']),
        'feasibility_ratings': len(ratings_df[ratings_df['RatingType'] == 'Feasibility'])
    }
    
    print(f"\n=== Summary Statistics ===")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 