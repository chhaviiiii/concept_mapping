#!/usr/bin/env python3
"""
Convert the existing concept mapping data to PyConceptMap format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def convert_data():
    """Convert the existing data to PyConceptMap format."""
    print("Converting data to PyConceptMap format...")
    
    # Load the original data
    data_path = Path('data/BCCS AI Workshop_August 11, 2025_23.45.csv')
    raw_data = pd.read_csv(data_path)
    
    print(f"Loaded {len(raw_data)} rows from original data")
    
    # 1. Create Statements.csv
    print("Creating Statements.csv...")
    statements = []
    for i in range(1, 101):  # 100 statements
        statements.append({
            'StatementID': i,
            'Statement': f'Statement {i}'
        })
    
    statements_df = pd.DataFrame(statements)
    statements_df.to_csv('data/Statements.csv', index=False)
    print(f"✅ Created Statements.csv with {len(statements)} statements")
    
    # 2. Create SortedCards.csv
    print("Creating SortedCards.csv...")
    sorting_cols = [col for col in raw_data.columns if col.startswith('Q1_')]
    print(f"Found {len(sorting_cols)} sorting columns")
    
    sorted_cards = []
    for participant_id, row in raw_data.iterrows():
        participant_sorting = {}
        
        # Extract sorting data for this participant
        for col in sorting_cols:
            statement_id = int(col.split('_')[1])
            group_value = str(row[col]).strip()
            if group_value and group_value != 'nan' and group_value != '':
                if group_value not in participant_sorting:
                    participant_sorting[group_value] = []
                participant_sorting[group_value].append(statement_id)
        
        # Create rows for each pile
        for pile_name, statements_in_pile in participant_sorting.items():
            row_data = {
                'SorterID': participant_id + 1,
                'PileName': pile_name
            }
            # Add statements to the row
            for i, stmt_id in enumerate(statements_in_pile):
                row_data[f'Statement_{i+1}'] = stmt_id
            
            sorted_cards.append(row_data)
    
    sorted_cards_df = pd.DataFrame(sorted_cards)
    sorted_cards_df.to_csv('data/SortedCards.csv', index=False)
    print(f"✅ Created SortedCards.csv with {len(sorted_cards)} sorting entries")
    
    # 3. Create Demographics.csv
    print("Creating Demographics.csv...")
    demographics = []
    for participant_id in range(len(raw_data)):
        demographics.append({
            'RaterID': participant_id + 1,
            'ParticipantID': participant_id + 1
        })
    
    demographics_df = pd.DataFrame(demographics)
    demographics_df.to_csv('data/Demographics.csv', index=False)
    print(f"✅ Created Demographics.csv with {len(demographics)} participants")
    
    # 4. Create Ratings.csv
    print("Creating Ratings.csv...")
    ratings = []
    
    # Extract importance ratings
    importance_cols = [col for col in raw_data.columns if col.startswith('Q2.1_')]
    feasibility_cols = [col for col in raw_data.columns if col.startswith('Q2.2_')]
    
    print(f"Found {len(importance_cols)} importance columns and {len(feasibility_cols)} feasibility columns")
    
    for participant_id, row in raw_data.iterrows():
        # Importance ratings
        for col in importance_cols:
            statement_id = int(col.split('_')[1])
            rating_text = str(row[col])
            rating_match = re.search(r'(\d+)\s*=', rating_text)
            if rating_match:
                rating = int(rating_match.group(1))
                ratings.append({
                    'RaterID': participant_id + 1,
                    'StatementID': statement_id,
                    'Importance': rating,
                    'Feasibility': np.nan  # Will be filled later
                })
        
        # Feasibility ratings
        for col in feasibility_cols:
            statement_id = int(col.split('_')[1])
            rating_text = str(row[col])
            rating_match = re.search(r'(\d+)\s*=', rating_text)
            if rating_match:
                rating = int(rating_match.group(1))
                # Find existing rating entry and update it
                for rating_entry in ratings:
                    if (rating_entry['RaterID'] == participant_id + 1 and 
                        rating_entry['StatementID'] == statement_id):
                        rating_entry['Feasibility'] = rating
                        break
                else:
                    # Create new entry if not found
                    ratings.append({
                        'RaterID': participant_id + 1,
                        'StatementID': statement_id,
                        'Importance': np.nan,
                        'Feasibility': rating
                    })
    
    ratings_df = pd.DataFrame(ratings)
    ratings_df.to_csv('data/Ratings.csv', index=False)
    print(f"✅ Created Ratings.csv with {len(ratings)} ratings")
    
    print("\n" + "="*60)
    print("✅ DATA CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Created files:")
    print("  - data/Statements.csv")
    print("  - data/SortedCards.csv") 
    print("  - data/Demographics.csv")
    print("  - data/Ratings.csv")
    print("\nYou can now run: python3 run_pyconceptmap.py --data_folder data")

if __name__ == '__main__':
    convert_data()
