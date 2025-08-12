import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ConceptMappingAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with the CSV file path."""
        self.csv_file = csv_file
        self.df = None
        self.statements = []
        self.grouping_data = None
        self.importance_data = None
        self.feasibility_data = None
        
    def load_data(self):
        """Load and preprocess the concept mapping data."""
        print("Loading concept mapping data...")
        
        # Read the CSV file
        self.df = pd.read_csv(self.csv_file)
        
        # Extract statement text from column headers
        statement_columns = [col for col in self.df.columns if col.startswith('Q1_')]
        self.statements = []
        
        for col in statement_columns:
            # Extract statement text from the long column description
            statement_text = col.split(' - ')[-1] if ' - ' in col else col
            self.statements.append(statement_text)
        
        print(f"Loaded {len(self.statements)} statements from {len(self.df)} participants")
        
        # Extract grouping data (Q1 columns)
        grouping_cols = [col for col in self.df.columns if col.startswith('Q1_')]
        self.grouping_data = self.df[grouping_cols].copy()
        
        # Extract importance ratings (Q2.1 columns)
        importance_cols = [col for col in self.df.columns if col.startswith('Q2.1_')]
        self.importance_data = self.df[importance_cols].copy()
        
        # Extract feasibility ratings (Q2.2 columns)
        feasibility_cols = [col for col in self.df.columns if col.startswith('Q2.2_')]
        self.feasibility_data = self.df[feasibility_cols].copy()
        
        # Clean column names
        self.grouping_data.columns = [f"Statement_{i+1}" for i in range(len(self.statements))]
        self.importance_data.columns = [f"Statement_{i+1}" for i in range(len(self.statements))]
        self.feasibility_data.columns = [f"Statement_{i+1}" for i in range(len(self.statements))]
        
        return self
    
    def analyze_grouping_patterns(self):
        """Analyze how participants grouped the statements."""
        print("\n=== GROUPING PATTERN ANALYSIS ===")
        
        # Count group assignments for each statement
        group_counts = defaultdict(lambda: defaultdict(int))
        
        for idx, row in self.grouping_data.iterrows():
            for col in self.grouping_data.columns:
                group = row[col]
                if pd.notna(group) and group != '':
                    group_counts[col][group] += 1
        
        # Find most common groups for each statement
        most_common_groups = {}
        for statement in self.grouping_data.columns:
            if statement in group_counts:
                most_common = max(group_counts[statement].items(), key=lambda x: x[1])
                most_common_groups[statement] = most_common
        
        print(f"Most common group assignments:")
        for statement, (group, count) in most_common_groups.items():
            statement_num = int(statement.split('_')[1])
            print(f"Statement {statement_num}: Group {group} ({count} participants)")
        
        return most_common_groups
    
    def analyze_importance_ratings(self):
        """Analyze importance ratings for each statement."""
        print("\n=== IMPORTANCE RATING ANALYSIS ===")
        
        # Convert text ratings to numeric values
        rating_map = {
            '1 = relatively unimportant (compared with the rest of the statements)': 1,
            '2 = somewhat important': 2,
            '3 = moderately important': 3,
            '4 = very important': 4,
            '5 = extremely important (compared with the rest of the statements)': 5
        }
        
        importance_numeric = self.importance_data.copy()
        for col in importance_numeric.columns:
            importance_numeric[col] = importance_numeric[col].map(rating_map)
        
        # Calculate mean importance for each statement
        mean_importance = importance_numeric.mean()
        
        print("Mean importance ratings (1-5 scale):")
        for statement, mean_rating in mean_importance.items():
            statement_num = int(statement.split('_')[1])
            print(f"Statement {statement_num}: {mean_rating:.2f}")
        
        # Find top 10 most important statements
        top_important = mean_importance.nlargest(10)
        print(f"\nTop 10 most important statements:")
        for statement, rating in top_important.items():
            statement_num = int(statement.split('_')[1])
            print(f"Statement {statement_num}: {rating:.2f}")
        
        return mean_importance
    
    def analyze_feasibility_ratings(self):
        """Analyze feasibility ratings for each statement."""
        print("\n=== FEASIBILITY RATING ANALYSIS ===")
        
        # Convert text ratings to numeric values
        rating_map = {
            '1 = relatively unfeasible (compared with the rest of the statements)': 1,
            '2 = somewhat feasible': 2,
            '3 = moderately feasible': 3,
            '4 = very feasible': 4,
            '5 = extremely feasible (compared with the rest of the statements)': 5
        }
        
        feasibility_numeric = self.feasibility_data.copy()
        for col in feasibility_numeric.columns:
            feasibility_numeric[col] = feasibility_numeric[col].map(rating_map)
        
        # Calculate mean feasibility for each statement
        mean_feasibility = feasibility_numeric.mean()
        
        print("Mean feasibility ratings (1-5 scale):")
        for statement, mean_rating in mean_feasibility.items():
            statement_num = int(statement.split('_')[1])
            print(f"Statement {statement_num}: {mean_rating:.2f}")
        
        # Find top 10 most feasible statements
        top_feasible = mean_feasibility.nlargest(10)
        print(f"\nTop 10 most feasible statements:")
        for statement, rating in top_feasible.items():
            statement_num = int(statement.split('_')[1])
            print(f"Statement {statement_num}: {rating:.2f}")
        
        return mean_feasibility
    
    def create_importance_feasibility_matrix(self):
        """Create importance-feasibility matrix for strategic planning."""
        print("\n=== IMPORTANCE-FEASIBILITY MATRIX ===")
        
        # Get numeric ratings
        importance_numeric = self.importance_data.copy()
        feasibility_numeric = self.feasibility_data.copy()
        
        rating_map = {
            '1 = relatively unimportant (compared with the rest of the statements)': 1,
            '2 = somewhat important': 2,
            '3 = moderately important': 3,
            '4 = very important': 4,
            '5 = extremely important (compared with the rest of the statements)': 5
        }
        
        feasibility_map = {
            '1 = relatively unfeasible (compared with the rest of the statements)': 1,
            '2 = somewhat feasible': 2,
            '3 = moderately feasible': 3,
            '4 = very feasible': 4,
            '5 = extremely feasible (compared with the rest of the statements)': 5
        }
        
        for col in importance_numeric.columns:
            importance_numeric[col] = importance_numeric[col].map(rating_map)
            feasibility_numeric[col] = feasibility_numeric[col].map(feasibility_map)
        
        mean_importance = importance_numeric.mean()
        mean_feasibility = feasibility_numeric.mean()
        
        # Create matrix
        matrix_data = pd.DataFrame({
            'Statement': [f"Statement_{i+1}" for i in range(len(self.statements))],
            'Importance': mean_importance.values,
            'Feasibility': mean_feasibility.values
        })
        
        # Categorize statements
        def categorize_statement(row):
            if row['Importance'] >= 4 and row['Feasibility'] >= 4:
                return 'High Priority - Easy'
            elif row['Importance'] >= 4 and row['Feasibility'] < 4:
                return 'High Priority - Hard'
            elif row['Importance'] < 4 and row['Feasibility'] >= 4:
                return 'Low Priority - Easy'
            else:
                return 'Low Priority - Hard'
        
        matrix_data['Category'] = matrix_data.apply(categorize_statement, axis=1)
        
        print("Strategic categorization:")
        for category in ['High Priority - Easy', 'High Priority - Hard', 'Low Priority - Easy', 'Low Priority - Hard']:
            statements = matrix_data[matrix_data['Category'] == category]
            print(f"\n{category} ({len(statements)} statements):")
            for _, row in statements.iterrows():
                statement_num = int(row['Statement'].split('_')[1])
                print(f"  Statement {statement_num}: Importance={row['Importance']:.2f}, Feasibility={row['Feasibility']:.2f}")
        
        return matrix_data
    
    def create_concept_map_visualization(self, matrix_data):
        """Create a visual concept map showing importance vs feasibility."""
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        colors = {
            'High Priority - Easy': 'green',
            'High Priority - Hard': 'red',
            'Low Priority - Easy': 'blue',
            'Low Priority - Hard': 'orange'
        }
        
        for category, color in colors.items():
            subset = matrix_data[matrix_data['Category'] == category]
            plt.scatter(subset['Feasibility'], subset['Importance'], 
                       c=color, label=category, s=100, alpha=0.7)
        
        # Add statement numbers as annotations
        for idx, row in matrix_data.iterrows():
            statement_num = int(row['Statement'].split('_')[1])
            plt.annotate(str(statement_num), 
                        (row['Feasibility'], row['Importance']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.xlabel('Feasibility Rating')
        plt.ylabel('Importance Rating')
        plt.title('Concept Map: AI in Cancer Care - Importance vs Feasibility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add quadrant lines
        plt.axhline(y=4, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=4, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('concept_map_importance_feasibility.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt
    
    def create_statement_summary(self):
        """Create a comprehensive summary of all statements with their ratings."""
        print("\n=== COMPREHENSIVE STATEMENT SUMMARY ===")
        
        # Get numeric ratings
        importance_numeric = self.importance_data.copy()
        feasibility_numeric = self.feasibility_data.copy()
        
        rating_map = {
            '1 = relatively unimportant (compared with the rest of the statements)': 1,
            '2 = somewhat important': 2,
            '3 = moderately important': 3,
            '4 = very important': 4,
            '5 = extremely important (compared with the rest of the statements)': 5
        }
        
        feasibility_map = {
            '1 = relatively unfeasible (compared with the rest of the statements)': 1,
            '2 = somewhat feasible': 2,
            '3 = moderately feasible': 3,
            '4 = very feasible': 4,
            '5 = extremely feasible (compared with the rest of the statements)': 5
        }
        
        for col in importance_numeric.columns:
            importance_numeric[col] = importance_numeric[col].map(rating_map)
            feasibility_numeric[col] = feasibility_numeric[col].map(feasibility_map)
        
        mean_importance = importance_numeric.mean()
        mean_feasibility = feasibility_numeric.mean()
        
        # Create summary dataframe
        summary_data = []
        for i in range(len(self.statements)):
            statement_num = i + 1
            statement_text = self.statements[i]
            importance = mean_importance.iloc[i]
            feasibility = mean_feasibility.iloc[i]
            
            summary_data.append({
                'Statement_Number': statement_num,
                'Statement_Text': statement_text,
                'Mean_Importance': importance,
                'Mean_Feasibility': feasibility,
                'Priority_Score': importance * feasibility  # Combined score
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Priority_Score', ascending=False)
        
        print("Top 20 statements by priority score (Importance × Feasibility):")
        for idx, row in summary_df.head(20).iterrows():
            print(f"\n{row['Statement_Number']}. {row['Statement_Text']}")
            print(f"   Importance: {row['Mean_Importance']:.2f}, Feasibility: {row['Mean_Feasibility']:.2f}")
            print(f"   Priority Score: {row['Priority_Score']:.2f}")
        
        return summary_df
    
    def run_complete_analysis(self):
        """Run the complete concept mapping analysis."""
        print("BCCS AI Workshop Concept Mapping Analysis")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Run analyses
        grouping_patterns = self.analyze_grouping_patterns()
        importance_ratings = self.analyze_importance_ratings()
        feasibility_ratings = self.analyze_feasibility_ratings()
        matrix_data = self.create_importance_feasibility_matrix()
        summary_df = self.create_statement_summary()
        
        # Create visualization
        self.create_concept_map_visualization(matrix_data)
        
        # Save results
        summary_df.to_csv('concept_mapping_summary.csv', index=False)
        matrix_data.to_csv('importance_feasibility_matrix.csv', index=False)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Files saved:")
        print("- concept_mapping_summary.csv: Complete statement summary")
        print("- importance_feasibility_matrix.csv: Strategic matrix data")
        print("- concept_map_importance_feasibility.png: Visual concept map")
        
        return {
            'grouping_patterns': grouping_patterns,
            'importance_ratings': importance_ratings,
            'feasibility_ratings': feasibility_ratings,
            'matrix_data': matrix_data,
            'summary_df': summary_df
        }

# Run the analysis
if __name__ == "__main__":
    analyzer = ConceptMappingAnalyzer('data/BCCS AI Workshop_August 11, 2025_23.45.csv')
    results = analyzer.run_complete_analysis() 