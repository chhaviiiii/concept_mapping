# User Guide

## Getting Started

This guide will walk you through using PyConceptMap for concept mapping analysis.

## Quick Start

### 1. Prepare Your Data

PyConceptMap requires four CSV files in your data folder:

- **Statements.csv**: List of statements with IDs
- **SortedCards.csv**: How participants sorted statements into piles
- **Demographics.csv**: Participant information
- **Ratings.csv**: Importance and feasibility ratings

### 2. Run Analysis

```bash
python run_pyconceptmap.py --data_folder /path/to/your/data
```

### 3. View Results

Check the output folder for:
- 8 visualization files (PNG)
- 7 report files (TXT/CSV)
- Statistical analysis results

## Data Preparation

### Statements.csv Format

```csv
StatementID,Statement
1,First statement text
2,Second statement text
3,Third statement text
```

**Requirements:**
- Must have `StatementID` and `Statement` columns
- StatementID must be sequential from 1 to N
- No missing values

### SortedCards.csv Format

```csv
SorterID,PileName,Statement_1,Statement_2,Statement_3
1,Pile A,1,2,3
1,Pile B,4,5
2,Pile 1,1,4
2,Pile 2,2,5
```

**Requirements:**
- Must have `SorterID` and `PileName` columns
- Additional columns contain statement IDs in each pile
- Each row represents one pile from one sorter

### Demographics.csv Format

```csv
RaterID,Age,Experience,Department
1,25,Low,A
2,30,Medium,B
3,35,High,A
```

**Requirements:**
- Must have `RaterID` column
- Additional columns for demographic variables
- RaterID should match those in Ratings.csv

### Ratings.csv Format

```csv
RaterID,StatementID,Importance,Feasibility
1,1,4,3
1,2,5,4
2,1,3,2
2,2,4,3
```

**Requirements:**
- Must have `RaterID`, `StatementID` columns
- Rating variables (e.g., Importance, Feasibility)
- All ratings should be on the same scale (e.g., 1-5)

## Command Line Usage

### Basic Commands

```bash
# Check if everything is installed correctly
python run_pyconceptmap.py --check_requirements

# Create sample data for testing
python run_pyconceptmap.py --create_sample_data

# Run analysis on your data
python run_pyconceptmap.py --data_folder ./my_data

# Run with custom output folder
python run_pyconceptmap.py --data_folder ./my_data --output_folder ./results
```

### Advanced Options

```bash
# Custom MDS method
python run_pyconceptmap.py --data_folder ./data --mds_method classical

# Custom clustering method
python run_pyconceptmap.py --data_folder ./data --clustering_method complete

# Specify number of clusters
python run_pyconceptmap.py --data_folder ./data --n_clusters 5

# Set random seed for reproducibility
python run_pyconceptmap.py --data_folder ./data --random_state 42

# Enable verbose output
python run_pyconceptmap.py --data_folder ./data --verbose
```

## Programmatic Usage

### Basic Analysis

```python
from pyconceptmap import ConceptMappingAnalysis

# Initialize analysis
analysis = ConceptMappingAnalysis(
    data_folder='./data',
    output_folder='./output',
    random_state=42
)

# Run complete analysis
success = analysis.run_complete_analysis()

if success:
    print("Analysis completed successfully!")
else:
    print("Analysis failed. Check the error messages.")
```

### Step-by-Step Analysis

```python
# Load data
analysis.load_data()

# Perform MDS
analysis.perform_mds(method='smacof')

# Perform clustering
analysis.perform_clustering(method='ward', auto_select=True)

# Analyze ratings
analysis.analyze_ratings()

# Generate visualizations
analysis.generate_visualizations()

# Generate reports
analysis.generate_reports()
```

### Custom Analysis

```python
# Custom parameters
analysis = ConceptMappingAnalysis('./data', random_state=123)

# Load and validate data
if not analysis.load_data():
    print("Data loading failed")
    exit(1)

# Custom MDS
analysis.perform_mds(method='classical')

# Custom clustering
analysis.perform_clustering(method='complete', n_clusters=3)

# Custom visualizations
analysis.visualizer.set_color_scheme('viridis')
analysis.generate_visualizations()

# Custom reports
analysis.generate_reports()
```

## Understanding Results

### Visualizations

1. **Point Map**: Shows MDS configuration with statement numbers
2. **Cluster Map**: Displays clusters with boundaries
3. **Point Rating Map**: Point sizes based on ratings
4. **Cluster Rating Map**: Cluster-level statistics
5. **Pattern Match**: Cluster comparison
6. **Go-Zone Plot**: Importance vs feasibility quadrants
7. **Dendrogram**: Hierarchical clustering tree
8. **Parallel Coordinates**: Multi-dimensional view

### Reports

1. **Sorter Summary**: Participant sorting behavior
2. **Rater Summary**: Demographics and statistics
3. **Statement Summary**: Individual statement statistics
4. **ANOVA Results**: Statistical significance tests
5. **Tukey's HSD**: Pairwise comparisons
6. **Cluster Analysis**: Cluster characteristics
7. **Comprehensive Report**: Complete summary

### Key Metrics

- **Stress Value**: MDS quality (lower is better)
- **Silhouette Score**: Cluster quality (higher is better)
- **F-statistic**: ANOVA significance
- **P-value**: Statistical significance
- **Cluster Sizes**: Number of statements per cluster

## Best Practices

### Data Quality

1. **Check for missing data** before analysis
2. **Validate data consistency** across files
3. **Look for outliers** in ratings
4. **Ensure proper data types** (numeric for ratings)

### Analysis Parameters

1. **Use default settings** for most analyses
2. **Try different clustering methods** if results seem unclear
3. **Check stress values** for MDS quality
4. **Validate cluster interpretation** with domain experts

### Interpretation

1. **Start with cluster maps** to understand groupings
2. **Use go-zone plots** for prioritization
3. **Check statistical significance** of differences
4. **Consider practical implications** of results

## Troubleshooting

### Common Issues

1. **File not found errors**: Check file paths and names
2. **Data format errors**: Verify CSV structure
3. **Memory errors**: Reduce dataset size or use sampling
4. **Convergence errors**: Try different MDS methods

### Getting Help

1. Check the [troubleshooting section](README.md#troubleshooting)
2. Review the [examples](examples/) folder
3. Open an [issue](https://github.com/pyconceptmap/pyconceptmap/issues)
4. Check the [documentation](README.md)

## Next Steps

After running your analysis:

1. **Review visualizations** to understand the concept map
2. **Check statistical reports** for significance
3. **Interpret clusters** in your domain context
4. **Use results** for decision-making or further research
5. **Share findings** with stakeholders

For more advanced usage, see the [API documentation](API.md) and [examples](examples/).
