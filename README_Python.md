# Python Concept Mapping Analysis

A comprehensive Python implementation of concept mapping analysis featuring multidimensional scaling, clustering analysis, and advanced visualizations.

## Overview

This Python implementation provides robust tools for concept mapping analysis, including data transformation, statistical analysis, and publication-quality visualizations. The analysis is designed for researchers conducting concept mapping studies in healthcare, education, business, and other domains.

## Features

### Core Analysis
- **Data Transformation**: Convert Qualtrics survey data to analysis format
- **Multidimensional Scaling (MDS)**: Reduce high-dimensional rating patterns to 2D visualization
- **K-means Clustering**: Optimal cluster selection using elbow method and silhouette analysis
- **Statistical Analysis**: Comprehensive correlation and descriptive statistics
- **Quality Metrics**: Cluster validation and goodness-of-fit measures

### Visualizations
- **Concept Maps**: 2D positioning with color-coded clusters
- **Importance vs Feasibility**: Strategic quadrant analysis
- **Rating Distributions**: Histograms and statistical summaries
- **Cluster Analysis**: Elbow plots and silhouette analysis
- **Similarity Heatmaps**: Correlation matrix visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

### Setup
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Verify installation
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, scipy; print('All packages installed successfully')"
```

## Usage

### Option 1: Use Existing Processed Data (Recommended)

The repository includes pre-processed data that works immediately:

```bash
# Run complete analysis
python3 concept_mapping_analysis_python.py
```

### Option 2: Transform New Data (Requires Customization)

For new Qualtrics exports, transformation may be required:

```bash
# Transform data (may need customization)
python3 transform_data_to_python.py

# Run analysis
python3 concept_mapping_analysis_python.py
```

### Complete Workflow

```bash
# Run the complete pipeline
python3 run_python_analysis.py
```

## Data Requirements

### Input Data Format

The analysis expects the following CSV files in `data/python_analysis/`:

**Statements.csv:**
```csv
StatementID,StatementText
1,"Human oversight during implementation in early days"
2,"Concerns with confidentiality"
```

**Ratings.csv:**
```csv
ParticipantID,StatementID,RatingType,Rating
P1,1,Importance,4
P1,1,Feasibility,3
P1,2,Importance,3
P1,2,Feasibility,4
```

**Demographics.csv (optional):**
```csv
ParticipantID,Age,Gender,Role,Experience
P1,35,Female,Physician,10
P2,28,Male,Resident,2
```

### Qualtrics Export Processing

**Important Note:** The transformation script requires customization for specific Qualtrics export formats. The current implementation handles:

- **Standard Qualtrics format**: Simple column names (Q1_1, Q2.1_1, Q2.2_1)
- **JSON format**: Complex column names with ImportId metadata

Customization may be needed for:
- Different question numbering schemes
- Varying header row structures
- Custom column naming conventions
- Specialized rating scales

**Working Data:** The repository includes pre-processed data in `data/rcmap_july27_2025/` that works without transformation.

## Analysis Process

### 1. Data Loading and Validation
- Load statements, ratings, and demographics
- Validate data structure and completeness
- Check for missing values and outliers
- Generate data quality reports

### 2. Similarity Matrix Creation
- Transform ratings into participant-statement matrix
- Calculate correlation-based similarity between statements
- Convert to distance matrix for MDS analysis
- Handle missing data and edge cases

### 3. Multidimensional Scaling (MDS)
- Apply classical MDS to reduce dimensionality
- Extract 2D coordinates for visualization
- Calculate stress value for goodness-of-fit
- Validate MDS solution quality

### 4. Cluster Analysis
- Determine optimal number of clusters using:
  - Elbow method (within-cluster sum of squares)
  - Silhouette analysis (cluster quality)
- Perform K-means clustering with optimal k
- Validate cluster assignments and quality

### 5. Statistical Analysis
- Calculate importance vs feasibility correlations
- Generate descriptive statistics by cluster
- Analyze rating distributions and patterns
- Create comprehensive summary reports

### 6. Visualization Generation
- Concept map with cluster coloring
- Importance vs feasibility scatter plot
- Rating distribution histograms
- Cluster analysis diagnostic plots
- Similarity matrix heatmap

## Output Files

### Visualizations (PNG format)
- `concept_map.png`: Main concept map with cluster assignments
- `importance_vs_feasibility.png`: Strategic quadrant analysis
- `rating_distribution.png`: Rating distribution histograms
- `cluster_analysis.png`: Elbow method and silhouette plots
- `similarity_heatmap.png`: Statement similarity matrix

### Data Files (CSV format)
- `summary_statistics.csv`: Comprehensive analysis results
- `statements_with_clusters.csv`: Final results with cluster assignments

### Analysis Results
- **Cluster assignments**: Each statement assigned to optimal cluster
- **MDS coordinates**: 2D positioning for visualization
- **Statistical summaries**: Correlations, means, standard deviations
- **Quality metrics**: Stress values, silhouette scores, cluster quality

## Configuration

### Analysis Parameters

**Clustering Parameters:**
```python
max_k = 10              # Maximum number of clusters to evaluate
nstart = 25             # Number of K-means initializations
random_state = 42       # Random seed for reproducibility
```

**MDS Parameters:**
```python
n_components = 2        # Number of dimensions for MDS
metric = True           # Use metric MDS (classical)
```

**Visualization Parameters:**
```python
figure_size = (12, 10)  # Figure dimensions in inches
dpi = 300              # Image resolution
color_palette = 'viridis'  # Color scheme for clusters
```

### Customization Options

**Modifying Cluster Analysis:**
```python
# Adjust maximum clusters
cluster_analysis = find_optimal_clusters(mds_coords, max_k=15)

# Force specific number of clusters
clustering_results = perform_clustering(mds_coords, n_clusters=3)
```

**Customizing Visualizations:**
```python
# Modify plot styles
plt.style.use('seaborn-v0_8')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 8)
```

## Troubleshooting

### Common Issues

**Data Format Errors:**
- Ensure CSV files have correct column names
- Check for missing values and data types
- Verify statement numbering consistency

**Memory Issues:**
- Reduce dataset size for large analyses
- Use data sampling for preliminary testing
- Optimize matrix operations for large datasets

**Visualization Problems:**
- Check matplotlib backend configuration
- Ensure output directory exists and is writable
- Verify color palette compatibility

### Data Quality Checks

**Before Analysis:**
```python
# Check data completeness
print(f"Statements: {len(statements)}")
print(f"Participants: {len(ratings['ParticipantID'].unique())}")
print(f"Ratings: {len(ratings)}")
print(f"Missing values: {ratings['Rating'].isna().sum()}")
```

**After Analysis:**
```python
# Validate results
print(f"Clusters: {len(set(cluster_labels))}")
print(f"MDS stress: {mds_results['stress']:.3f}")
print(f"Correlation: {correlation:.3f}")
```

## Performance Optimization

### Large Datasets
- Use efficient matrix operations with numpy
- Implement data sampling for preliminary analysis
- Optimize memory usage with data types

### Computational Efficiency
- Vectorize operations where possible
- Use efficient clustering algorithms
- Optimize visualization rendering

## Extending the Analysis

### Adding New Metrics
```python
def calculate_additional_metrics(ratings, statements):
    # Add custom statistical measures
    pass
```

### Custom Visualizations
```python
def create_custom_plot(data, output_path):
    # Create specialized visualizations
    pass
```

### Integration with Other Tools
- Export results for external analysis
- Generate reports in different formats
- Interface with other statistical software

## Best Practices

### Data Preparation
- Validate data quality before analysis
- Document data transformation steps
- Maintain consistent naming conventions

### Analysis Workflow
- Use reproducible random seeds
- Document parameter choices
- Validate results with multiple methods

### Result Interpretation
- Consider context when interpreting clusters
- Validate findings with domain experts
- Document limitations and assumptions

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review code comments for implementation details
3. Validate data format and quality
4. Test with sample data first

## License

Educational and Research Use

## Acknowledgments

Developed for concept mapping research in healthcare and related domains. This implementation supports the analysis of complex conceptual frameworks and strategic planning processes. 