import pandas as pd
import numpy as np
import os
from collections import defaultdict
import re

def process_august11_data():
    """Process the August 11 raw data and convert it to the same format as July 27 data."""
    
    # Create output directory
    output_dir = "data/rcmap_august11_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the raw August 11 data
    raw_data = pd.read_csv('data/BCCS AI Workshop_August 11, 2025_23.45.csv')
    
    print(f"Processing August 11 data with {len(raw_data)} participants")
    
    # Extract statements from column headers
    statement_columns = [col for col in raw_data.columns if col.startswith('Q1_')]
    statements = []
    
    for col in statement_columns:
        # Extract statement text from the long column description
        statement_text = col.split(' - ')[-1] if ' - ' in col else col
        statements.append(statement_text)
    
    print(f"Found {len(statements)} statements")
    
    # Create Statements.csv
    statements_df = pd.DataFrame({
        'StatementID': range(1, len(statements) + 1),
        'StatementText': statements
    })
    statements_df.to_csv(f'{output_dir}/Statements.csv', index=False)
    print(f"Created {output_dir}/Statements.csv")
    
    # Process ratings data
    ratings_data = []
    
    # Extract importance ratings (Q2.1 columns)
    importance_cols = [col for col in raw_data.columns if col.startswith('Q2.1_')]
    feasibility_cols = [col for col in raw_data.columns if col.startswith('Q2.2_')]
    
    # Function to extract numeric rating from text
    def extract_rating(rating_text):
        if pd.isna(rating_text) or rating_text == '':
            return None
        
        # Look for pattern like "4 = very important" or "3 = moderately feasible"
        match = re.search(r'(\d+)\s*=', str(rating_text))
        if match:
            return int(match.group(1))
        return None
    
    # Process each participant
    for participant_idx, row in raw_data.iterrows():
        participant_id = participant_idx + 1
        
        # Process importance ratings
        for i, col in enumerate(importance_cols):
            statement_id = i + 1
            rating_text = row[col]
            rating_value = extract_rating(rating_text)
            
            if rating_value is not None:
                ratings_data.append({
                    'ParticipantID': participant_id,
                    'StatementID': statement_id,
                    'Rating': rating_value,
                    'RatingType': 'Importance'
                })
        
        # Process feasibility ratings
        for i, col in enumerate(feasibility_cols):
            statement_id = i + 1
            rating_text = row[col]
            rating_value = extract_rating(rating_text)
            
            if rating_value is not None:
                ratings_data.append({
                    'ParticipantID': participant_id,
                    'StatementID': statement_id,
                    'Rating': rating_value,
                    'RatingType': 'Feasibility'
                })
    
    # Create Ratings.csv
    ratings_df = pd.DataFrame(ratings_data)
    ratings_df.to_csv(f'{output_dir}/Ratings.csv', index=False)
    print(f"Created {output_dir}/Ratings.csv with {len(ratings_df)} ratings")
    
    # Process grouping data
    grouping_data = []
    grouping_cols = [col for col in raw_data.columns if col.startswith('Q1_')]
    
    for participant_idx, row in raw_data.iterrows():
        participant_id = participant_idx + 1
        
        for i, col in enumerate(grouping_cols):
            statement_id = i + 1
            group = row[col]
            
            if pd.notna(group) and group != '':
                # Clean up group name
                if isinstance(group, str):
                    group = group.strip()
                
                grouping_data.append({
                    'ParticipantID': participant_id,
                    'StatementID': statement_id,
                    'PileID': group
                })
    
    # Create SortedCards.csv
    sorted_cards_df = pd.DataFrame(grouping_data)
    sorted_cards_df.to_csv(f'{output_dir}/SortedCards.csv', index=False)
    print(f"Created {output_dir}/SortedCards.csv with {len(sorted_cards_df)} groupings")
    
    # Create Demographics.csv
    demographics_data = []
    
    for participant_idx, row in raw_data.iterrows():
        participant_id = participant_idx + 1
        
        # Extract demographic information
        name = row.get('Q6', '')  # Name field
        role = row.get('Q7', '')  # Role field
        
        demographics_data.append({
            'ParticipantID': participant_id,
            'Name': name if pd.notna(name) else '',
            'Role': role if pd.notna(role) else '',
            'ResponseDate': row.get('RecordedDate', ''),
            'Duration': row.get('Duration (in seconds)', ''),
            'Progress': row.get('Progress', '')
        })
    
    # Create Demographics.csv
    demographics_df = pd.DataFrame(demographics_data)
    demographics_df.to_csv(f'{output_dir}/Demographics.csv', index=False)
    print(f"Created {output_dir}/Demographics.csv")
    
    # Print summary statistics
    print("\n=== AUGUST 11 DATA PROCESSING SUMMARY ===")
    print(f"Total participants: {len(raw_data)}")
    print(f"Total statements: {len(statements)}")
    print(f"Total importance ratings: {len(ratings_df[ratings_df['RatingType'] == 'Importance'])}")
    print(f"Total feasibility ratings: {len(ratings_df[ratings_df['RatingType'] == 'Feasibility'])}")
    print(f"Total groupings: {len(sorted_cards_df)}")
    
    # Check for missing data
    expected_importance = len(raw_data) * len(statements)
    expected_feasibility = len(raw_data) * len(statements)
    actual_importance = len(ratings_df[ratings_df['RatingType'] == 'Importance'])
    actual_feasibility = len(ratings_df[ratings_df['RatingType'] == 'Feasibility'])
    
    print(f"Importance rating completion rate: {actual_importance/expected_importance:.2%}")
    print(f"Feasibility rating completion rate: {actual_feasibility/expected_feasibility:.2%}")
    
    # Show some sample ratings
    print(f"\nSample importance ratings:")
    sample_importance = ratings_df[ratings_df['RatingType'] == 'Importance'].head(10)
    for _, row in sample_importance.iterrows():
        print(f"  Participant {row['ParticipantID']}, Statement {row['StatementID']}: {row['Rating']}")
    
    print(f"\nSample feasibility ratings:")
    sample_feasibility = ratings_df[ratings_df['RatingType'] == 'Feasibility'].head(10)
    for _, row in sample_feasibility.iterrows():
        print(f"  Participant {row['ParticipantID']}, Statement {row['StatementID']}: {row['Rating']}")
    
    return {
        'statements_df': statements_df,
        'ratings_df': ratings_df,
        'sorted_cards_df': sorted_cards_df,
        'demographics_df': demographics_df
    }

if __name__ == "__main__":
    results = process_august11_data()
    print("\nData processing complete! Files saved to data/rcmap_august11_2025/") 