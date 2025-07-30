# Concept Mapping Analysis Toolkit

A comprehensive toolkit for conducting concept mapping analysis using both Python and R implementations. This repository provides researchers with robust tools for analyzing qualitative and quantitative data to identify patterns, relationships, and clusters in complex conceptual frameworks.

## Overview

Concept mapping is a research methodology that combines qualitative and quantitative approaches to:
- Generate and structure ideas
- Identify relationships between concepts
- Create visual representations of complex data
- Support decision-making and strategic planning

This toolkit is designed for researchers in healthcare, education, business, and other domains requiring structured analysis of complex ideas and their relationships.

## Project Structure

```
RCMap-1/
├── data/
│   ├── BCCS AI Workshop_July 27, 2025_15.23.csv    # Original Qualtrics export
│   ├── rcmap_july27_2025/                          # Working processed data
│   ├── python_analysis/                            # Python analysis data
│   └── rcmap_analysis/                             # R analysis data
├── Figures/
│   ├── analysis/                                   # R analysis outputs
│   └── python_analysis/                            # Python analysis outputs
├── concept_mapping_analysis_python.py              # Core Python analysis
├── concept_mapping_analysis.R                      # Core R analysis
├── transform_data_to_python.py                     # Python data transformation
├── transform_data_to_rcmap.R                       # R data transformation
├── run_python_analysis.py                          # Python workflow script
├── run_analysis.R                                  # R workflow script
├── requirements.R                                  # R package dependencies
├── README_Python.md                                # Python documentation
├── README_R.md                                     # R documentation
└── README.md                                       # This file
```

## Quick Start

### Prerequisites

**Python Requirements:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**R Requirements:**
```bash
Rscript requirements.R
```

### Running the Analysis

**Option 1: Use Existing Processed Data (Recommended)**
```bash
# Run R analysis
Rscript concept_mapping_analysis.R

# Run Python analysis  
python3 concept_mapping_analysis_python.py
```

**Option 2: Transform New Data (Requires Customization)**
```bash
# Transform data for R analysis
Rscript transform_data_to_rcmap.R

# Transform data for Python analysis
python3 transform_data_to_python.py
```

## Data Transformation

**Important Note:** The data transformation scripts require customization for specific Qualtrics export formats. The current scripts are configured for the dataset `data/BCCS AI Workshop_July 27, 2025_15.23.csv` but may need adjustments for different export formats.

**Working Data:** The repository includes pre-processed data in `data/rcmap_july27_2025/` that works with both R and Python analyses without transformation.

### Qualtrics Export Formats

The transformation scripts handle different Qualtrics export formats:
- **Standard format**: Simple column names (Q1_1, Q2.1_1, Q2.2_1)
- **JSON format**: Complex column names with ImportId metadata

Customization may be required for:
- Different question numbering schemes
- Varying header row structures
- Custom column naming conventions
- Specialized rating scales

## Analysis Features

### Core Analysis Components
- **Multidimensional Scaling (MDS)**: Reduces high-dimensional data to 2D visualization
- **K-means Clustering**: Groups similar statements using optimal cluster selection
- **Similarity Matrix**: Calculates correlations between statement rating patterns
- **Statistical Analysis**: Comprehensive summary statistics and validation

### Visualization Outputs
- **Concept Maps**: 2D positioning of statements with cluster coloring
- **Importance vs Feasibility Plots**: Strategic quadrant analysis
- **Rating Distributions**: Histograms of importance and feasibility ratings
- **Cluster Analysis**: Elbow method and silhouette analysis plots
- **Similarity Heatmaps**: Correlation matrices between statements

### Output Files
- `concept_map.png`: Main concept map visualization
- `importance_vs_feasibility.png`: Strategic quadrant plot
- `rating_distribution.png`: Rating distribution histograms
- `cluster_analysis.png`: Cluster selection analysis
- `similarity_heatmap.png`: Statement similarity matrix
- `summary_statistics.csv`: Comprehensive results summary
- `statements_with_clusters.csv`: Final results with cluster assignments

## Implementation Details

### Python Implementation
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
- **Analysis**: MDS, K-means clustering, statistical analysis
- **Output**: PNG visualizations and CSV results

### R Implementation  
- **Libraries**: dplyr, ggplot2, cluster, factoextra, MASS, corrplot
- **Analysis**: MDS, K-means clustering, statistical analysis
- **Output**: PNG visualizations and CSV results

## Results Interpretation

### Cluster Analysis
- **Optimal Clusters**: Automatically determined using elbow method and silhouette analysis
- **Cluster Characteristics**: Each cluster represents statements with similar rating patterns
- **Interpretation**: Clusters can be interpreted as conceptual themes or strategic areas

### Strategic Quadrants
- **High Importance, High Feasibility**: Priority implementation areas
- **High Importance, Low Feasibility**: Strategic challenges requiring attention
- **Low Importance, High Feasibility**: Quick wins with limited impact
- **Low Importance, Low Feasibility**: Low priority areas

### Correlation Analysis
- **Importance vs Feasibility**: Measures relationship between perceived importance and implementation feasibility
- **Interpretation**: High correlation suggests alignment, low correlation indicates strategic tensions

## Customization

### Modifying Analysis Parameters
- **Number of Clusters**: Adjust `max_k` parameter in clustering functions
- **MDS Dimensions**: Modify `n_components` parameter for different dimensionality
- **Visualization Styles**: Customize plot parameters in visualization functions

### Adding New Analysis Components
- **Additional Metrics**: Extend statistical analysis functions
- **Custom Visualizations**: Add new plotting functions
- **Data Validation**: Implement additional quality checks

## Troubleshooting

### Common Issues
- **Data Format**: Ensure Qualtrics export matches expected format
- **Missing Dependencies**: Install required packages for Python or R
- **File Paths**: Verify data files are in correct directories
- **Memory Issues**: Reduce dataset size for large analyses

### Data Quality Checks
- **Missing Values**: Check for incomplete rating data
- **Outliers**: Identify and handle extreme rating values
- **Consistency**: Verify statement numbering and participant IDs

## Contributing

This toolkit is designed for educational and research use. Contributions are welcome for:
- Additional analysis methods
- Enhanced visualizations
- Improved data transformation
- Documentation improvements

## License

Educational and Research Use

## Acknowledgments

Developed for concept mapping research in healthcare and related domains. This toolkit supports the analysis of complex conceptual frameworks and strategic planning processes.
