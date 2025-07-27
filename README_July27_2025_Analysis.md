# BCCS AI Workshop July 27, 2025 Concept Mapping Analysis

This repository contains a comprehensive concept mapping analysis program for the BCCS AI Workshop data collected on July 27, 2025. The analysis covers 100 statements about AI in cancer care, including grouping patterns and importance/feasibility ratings.

## Overview

The concept mapping analysis reveals how participants conceptualize and prioritize different aspects of AI in cancer care. The program processes:
- **100 statements** about AI in cancer care
- **Grouping data** (how participants organized statements into meaningful groups)
- **Rating data** (importance and feasibility ratings for each statement)
- **Participant demographics** and metadata

## Files Structure

```
├── transform_july27_2025_to_rcmap.R    # Data transformation script
├── concept_mapping_analysis.R           # Main analysis script
├── run_july27_2025_analysis.R          # Master script (runs everything)
├── data/
│   ├── BCCS AI Workshop_July 27, 2025_15.23.csv    # Raw CSV data
│   ├── BCCS AI Workshop_July 27, 2025_15.26.tsv    # Raw TSV data
│   ├── BCCS AI Workshop_July 27, 2025_15.26.xlsx   # Raw Excel data
│   └── rcmap_july27_2025/              # Transformed data (generated)
│       ├── Statements.csv              # 100 statements with IDs
│       ├── SortedCards.csv             # Grouping data
│       ├── Ratings.csv                 # Importance/feasibility ratings
│       └── Demographics.csv            # Participant metadata
└── Figures/july27_2025_analysis/       # Output visualizations (generated)
    ├── concept_map.png                 # Main concept map
    ├── importance_vs_feasibility.png   # Rating comparison
    ├── rating_distribution.png         # Rating distributions
    ├── cluster_summary.csv             # Cluster analysis results
    ├── importance_feasibility_summary.csv # Rating summaries
    └── analysis_report.html            # Comprehensive HTML report
```

## Quick Start

### Prerequisites

Install required R packages:

```r
install.packages(c("dplyr", "readr", "ggplot2", "ggrepel", "cluster", 
                   "factoextra", "MASS", "corrplot", "viridis", 
                   "RColorBrewer", "gridExtra", "knitr", "kableExtra", 
                   "tidyr", "stringr", "purrr", "data.table", "readxl"))
```

### Run Complete Analysis

Execute the master script to run the entire pipeline:

```r
source("run_july27_2025_analysis.R")
```

This will:
1. Transform all July 27, 2025 data files to RCMap format
2. Perform concept mapping analysis
3. Generate visualizations and reports

### Run Individual Steps

**Step 1: Transform Data**
```r
source("transform_july27_2025_to_rcmap.R")
```

**Step 2: Perform Analysis**
```r
source("concept_mapping_analysis.R")
```

## Analysis Components

### 1. Data Transformation
The transformation script processes:
- **CSV files**: Qualtrics export format
- **TSV files**: Tab-separated format
- **Excel files**: XLSX format

Extracts:
- 100 statements from Q1 columns
- Grouping data (how participants sorted statements)
- Importance ratings (Q2.1 columns)
- Feasibility ratings (Q2.2 columns)
- Participant metadata

### 2. Concept Mapping Analysis
The analysis includes:

#### Grouping Analysis
- **Adjacency matrices**: How often statements were grouped together
- **Distance matrix**: Similarity between statements based on grouping patterns
- **Multidimensional Scaling (MDS)**: 2D representation of statement relationships
- **K-means clustering**: Optimal grouping of statements into conceptual clusters

#### Rating Analysis
- **Importance ratings**: How important each statement is (1-5 scale)
- **Feasibility ratings**: How feasible each statement is to implement (1-5 scale)
- **Correlation analysis**: Relationship between importance and feasibility
- **Statistical summaries**: Mean, median, standard deviation for each statement

### 3. Visualizations

#### Concept Map
- 2D scatter plot showing statement relationships
- Color-coded clusters
- Statement IDs labeled for easy reference

#### Importance vs Feasibility Plot
- Scatter plot comparing importance and feasibility
- Mean lines for reference
- Identifies high-priority, high-feasibility statements

#### Rating Distributions
- Histograms showing distribution of ratings
- Separate plots for importance and feasibility

#### Cluster Analysis Plots
- Elbow plot (WSS) for optimal cluster determination
- Silhouette plot for cluster quality assessment
- Gap statistic plot for cluster validation

### 4. Reports

#### CSV Reports
- `cluster_summary.csv`: All statements with their assigned clusters
- `cluster_ratings.csv`: Average ratings by cluster
- `importance_feasibility_summary.csv`: Combined rating data

#### HTML Report
- Comprehensive analysis report with:
  - Executive summary
  - Cluster descriptions
  - Top statements by importance and feasibility
  - Statistical summaries
  - Interactive elements

## Key Findings

The analysis reveals:

1. **Conceptual Clusters**: How participants naturally group AI in cancer care concepts
2. **Priority Areas**: Which statements are rated highest for importance
3. **Implementation Readiness**: Which statements are rated highest for feasibility
4. **Gap Analysis**: Areas that are important but not feasible (or vice versa)
5. **Consensus Patterns**: How much agreement exists among participants

## Customization

### Modify Analysis Parameters

Edit `concept_mapping_analysis.R` to adjust:
- Number of clusters (currently auto-determined)
- MDS dimensions (currently 2D)
- Visualization styles and colors
- Report formats

### Add New Data Sources

To analyze additional data files:
1. Add file paths to `input_files` in `transform_july27_2025_to_rcmap.R`
2. Ensure files follow the same Qualtrics format
3. Run the transformation script

### Custom Visualizations

The analysis generates ggplot2 objects that can be modified:
```r
# Modify concept map
concept_map + 
  theme_dark() +
  scale_color_brewer(palette = "Set1")

# Modify importance vs feasibility plot
importance_feasibility_plot +
  geom_smooth(method = "lm") +
  facet_wrap(~Cluster)
```

## Troubleshooting

### Common Issues

1. **Missing packages**: Install all required packages listed in prerequisites
2. **File not found**: Ensure all July 27, 2025 data files are in the `data/` directory
3. **Memory issues**: For large datasets, consider processing files individually
4. **Encoding issues**: Ensure CSV files use UTF-8 encoding

### Data Quality Checks

The scripts include validation for:
- Missing data
- Inconsistent participant IDs
- Invalid ratings (outside 1-5 range)
- Duplicate statements

### Performance Optimization

For large datasets:
- Process files individually rather than combining
- Reduce number of clusters for faster analysis
- Use sampling for preliminary analysis

## Output Interpretation

### Concept Map
- **Closer points**: Statements frequently grouped together
- **Distant points**: Statements rarely grouped together
- **Clusters**: Thematically related groups of statements

### Rating Analysis
- **High importance, high feasibility**: Priority implementation areas
- **High importance, low feasibility**: Research/development priorities
- **Low importance, high feasibility**: Quick wins
- **Low importance, low feasibility**: Lower priority areas

### Statistical Significance
- **Correlation coefficient**: Strength of relationship between importance and feasibility
- **Cluster stability**: How well-defined the conceptual groups are
- **Participant agreement**: Consensus on statement groupings and ratings

## Contact and Support

For questions about the analysis or to request modifications:
- Review the code comments for detailed explanations
- Check the generated HTML report for comprehensive results
- Examine the CSV outputs for detailed data

## License

This analysis program is designed for research purposes. Please ensure proper attribution when using the results or methodology. 