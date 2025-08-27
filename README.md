# BCCS AI Workshop - Concept Mapping Analysis

A comprehensive concept mapping analysis tool for the BCCS AI Workshop August 11, 2025 data. This tool generates 17 different visualizations and provides organized data outputs for strategic decision-making.

## 🎯 Overview

This project performs concept mapping analysis on workshop data to identify strategic priorities, cluster statements, and provide actionable insights through multiple visualization techniques.

## 📁 Project Structure

```
RCMap-1/
├── data/
│   └── BCCS AI Workshop_August 11, 2025_23.45.csv    # Raw workshop data
├── concept_mapping_analysis.py                       # Main analysis script
├── run_concept_mapping.py                           # Simple runner script
├── requirements.txt                                  # Python dependencies
└── README.md                                        # This file
```

## 🚀 Quick Start

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Run Analysis

Execute the complete analysis with a single command:

```bash
python run_concept_mapping.py
```

## 📊 Using This Analysis for Your Own Data

### Data Format Requirements

To use this analysis with your own data, your CSV file should follow this structure:

#### Required Columns:
- **Statement Grouping**: `Q1_1`, `Q1_2`, `Q1_3`, ... `Q1_N` (where N = number of statements)
- **Importance Ratings**: `Q2.1_1`, `Q2.1_2`, `Q2.1_3`, ... `Q2.1_N`
- **Feasibility Ratings**: `Q2.2_1`, `Q2.2_2`, `Q2.2_3`, ... `Q2.2_N`
- **Optional**: Participant information columns (Q6, Q7, etc.)

#### Rating Format:
Ratings should be in text format like:
- "4 = very important"
- "3 = somewhat important" 
- "2 = not very important"
- "1 = not important at all"

### Step-by-Step Guide for Your Data

#### 1. Prepare Your Data
```bash
# Create a data directory
mkdir data

# Place your CSV file in the data directory
# Rename it to match the expected format or update the script
```

#### 2. Update the Data Path (if needed)
Edit `run_concept_mapping.py` and change line 18:
```python
# Change this line to your data file name
data_path = Path("data/YOUR_DATA_FILE.csv")
```

#### 3. Run the Analysis
```bash
python3 run_concept_mapping.py
```

### Customizing for Different Data Formats

#### Option 1: Modify the Script
If your data has different column names, edit `concept_mapping_analysis.py`:

```python
# In the load_data method, update these lines:
importance_cols = [col for col in raw_data.columns if col.startswith('YOUR_IMPORTANCE_PREFIX')]
feasibility_cols = [col for col in raw_data.columns if col.startswith('YOUR_FEASIBILITY_PREFIX')]
```

#### Option 2: Rename Your Columns
Rename your CSV columns to match the expected format:
- `Q2.1_1`, `Q2.1_2`, ... for importance ratings
- `Q2.2_1`, `Q2.2_2`, ... for feasibility ratings

#### Option 3: Create a Custom Data Loader
For completely different data formats, create a new method in the `ConceptMappingAnalysis` class:

```python
def load_custom_data(self, data_path):
    """Load data in your custom format."""
    # Your custom data loading logic here
    pass
```

### Data Requirements Checklist

Before running the analysis, ensure your data has:

- [ ] **100 statements** (or modify the script for different numbers)
- [ ] **Importance ratings** for each statement
- [ ] **Feasibility ratings** for each statement  
- [ ] **Multiple participants** (recommended: 10+ participants)
- [ ] **Consistent rating scale** (1-4 or 1-5 scale)
- [ ] **No missing values** (or handle them appropriately)

### Example Data Structure

Your CSV should look like this:

```csv
ParticipantID,Q1_1,Q1_2,Q1_3,...,Q2.1_1,Q2.1_2,Q2.1_3,...,Q2.2_1,Q2.2_2,Q2.2_3,...
1,Group1,Group1,Group2,...,"4 = very important","3 = somewhat important","2 = not very important",...,"3 = somewhat feasible","4 = very feasible","2 = not very feasible",...
2,Group2,Group1,Group3,...,"3 = somewhat important","4 = very important","1 = not important",...,"4 = very feasible","3 = somewhat feasible","1 = not feasible",...
```

### Troubleshooting Your Data

#### Common Issues and Solutions:

1. **"KeyError: 'RatingType'"**
   - **Solution**: Check that your rating columns follow the expected naming pattern

2. **"No numeric rating found"**
   - **Solution**: Ensure ratings are in text format like "4 = very important"

3. **"Data file not found"**
   - **Solution**: Check the file path in `run_concept_mapping.py`

4. **"ValueError: x and y must have same first dimension"**
   - **Solution**: Ensure you have the same number of importance and feasibility ratings

### Advanced Customization

#### Changing Analysis Parameters
Edit `concept_mapping_analysis.py` to modify:

```python
# Number of clusters to test
for k in range(2, 6):  # Change 6 to test more clusters

# MDS parameters
mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')

# Clustering method
clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
```

#### Adding New Visualizations
To add custom figures:

```python
def create_custom_figure(self):
    """Add your custom visualization here."""
    # Your visualization code
    plt.savefig(self.figures_dir / 'custom_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
```

### Best Practices

1. **Data Quality**: Ensure clean, consistent data before analysis
2. **Sample Size**: Aim for 10+ participants for reliable results
3. **Rating Scale**: Use consistent 1-4 or 1-5 scales
4. **Statement Count**: 50-200 statements work best for concept mapping
5. **Backup**: Keep original data files before processing

## 📊 Output Structure

The analysis creates an organized output directory:

```
concept_mapping_output/
├── figures/                    # All 17 generated visualizations
│   ├── figure_1_importance_feasibility_scatter.png
│   ├── figure_2_quadrant_analysis.png
│   ├── figure_3_bubble_chart.png
│   ├── figure_4_optimal_cluster_analysis.png
│   ├── figure_5_cluster_comparison.png
│   ├── figure_6_radar_chart.png
│   ├── figure_7_statement_heatmap.png
│   ├── figure_8_grouping_frequency.png
│   ├── figure_9_gap_analysis.png
│   ├── figure_10_strategic_priorities.png
│   ├── figure_11_slope_graph.png
│   ├── figure_12_point_map.png
│   ├── figure_13_cluster_map.png
│   ├── figure_14_point_rating_map.png
│   ├── figure_15_cluster_rating_map.png
│   ├── figure_16_pattern_match.png
│   └── figure_17_go_zone_plot.png
├── processed_data/             # Analysis data files
│   ├── importance_feasibility_matrix.csv
│   ├── cluster_summary.csv
│   ├── cluster_ratings.csv
│   └── mds_coordinates.csv
└── reports/                    # Analysis reports
    └── analysis_summary.txt
```

## 📈 Generated Figures

### Basic Analysis (Figures 1-3)
1. **Importance vs Feasibility Scatter Plot** - Shows all statements with four quadrants
2. **Enhanced Quadrant Analysis** - Point size = importance, colors = quadrant
3. **Bubble Chart** - Bubble size = importance, color = gap

### Cluster Analysis (Figures 4-5, 12-13)
4. **Optimal Cluster Analysis** - Elbow method and silhouette analysis
5. **Cluster Comparison** - Mean ratings by cluster
12. **Point Map (MDS)** - Multidimensional scaling visualization
13. **Cluster Map** - Shows clusters with convex hull boundaries

### Advanced Visualizations (Figures 6-11, 14-17)
6. **Radar Chart** - Top 5 most important statements
7. **Statement Performance Heatmap** - Participant ratings heatmaps
8. **Grouping Frequency Analysis** - Frequency of statement groupings
9. **Gap Analysis** - Bar chart of importance-feasibility gaps
10. **Strategic Priorities** - Top 3 statements per category
11. **Slope Graph** - Cluster-level ratings comparison
14. **Point Rating Map** - MDS with importance ratings
15. **Cluster Rating Map** - Cluster-level importance ratings
16. **Pattern Match** - Parallel coordinates with correlation
17. **Go-Zone Plot** - Above/below average statements

## 📋 Data Files

### importance_feasibility_matrix.csv
Contains the main analysis data with:
- StatementID
- mean_Importance, std_Importance, count_Importance
- mean_Feasibility, std_Feasibility, count_Feasibility
- Gap (Importance - Feasibility)
- Cluster assignment

### cluster_summary.csv
Individual statement assignments with:
- StatementID
- Importance_Mean, Feasibility_Mean
- Gap
- Cluster

### cluster_ratings.csv
Aggregated cluster statistics with:
- Cluster ID
- Mean, standard deviation, and count for importance and feasibility
- Mean and standard deviation for gaps

### mds_coordinates.csv
Multidimensional scaling coordinates:
- StatementID
- Dimension1, Dimension2

## 🔍 Analysis Methods

### Clustering
- **Method**: Ward's Hierarchical Clustering
- **Optimal Clusters**: Determined using silhouette analysis
- **Range Tested**: 2-5 clusters

### Multidimensional Scaling (MDS)
- **Dimensions**: 2D
- **Distance Metric**: Euclidean distance
- **Stress**: Calculated for fit quality

### Statistical Analysis
- **Importance-Feasibility Gap**: Calculated for each statement
- **Quadrant Analysis**: Based on median values
- **Strategic Categories**: Immediate Implementation, Research & Development, Quick Wins

## 🎨 Visualization Features

- **High Resolution**: All figures saved at 300 DPI
- **Professional Styling**: Consistent color scheme and formatting
- **Publication Ready**: Suitable for reports and presentations
- **Accessible**: Good contrast and clear labeling

## 📊 Key Findings

### Data Summary
- **Total Statements**: 100
- **Optimal Clusters**: 2 (determined statistically)
- **Analysis Method**: Concept Mapping with MDS and Hierarchical Clustering

### Strategic Insights
- **High Priority & High Feasibility**: 32 statements (immediate implementation)
- **Research & Development**: 17 statements (high importance, low feasibility)
- **Quick Wins**: 12 statements (low importance, high feasibility)
- **Low Priority**: 39 statements (low importance, low feasibility)

## 🛠️ Technical Details

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

### Performance
- **Runtime**: ~30-60 seconds
- **Memory Usage**: Minimal (efficient design)
- **Output Size**: ~5MB total

## 🔧 Customization

### Modifying Analysis Parameters
Edit the `concept_mapping_analysis.py` file to:
- Change clustering method
- Adjust MDS parameters
- Modify color schemes
- Add new visualizations

### Adding New Figures
1. Add a new method to the `ConceptMappingAnalysis` class
2. Call it in the `create_all_figures()` method
3. Update the documentation

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data File Not Found**
   - Ensure `data/BCCS AI Workshop_August 11, 2025_23.45.csv` exists
   - Check file permissions

3. **Memory Issues**
   - The script is designed to be memory-efficient
   - Close other applications if needed

### Error Messages

- **"Data file not found"**: Check the data directory and file name
- **Import errors**: Install missing packages with pip
- **Permission errors**: Check write permissions for output directory

## 📝 Usage Examples

### Basic Analysis
```python
from concept_mapping_analysis import ConceptMappingAnalysis

# Initialize analysis
analysis = ConceptMappingAnalysis()

# Run complete analysis
analysis.run_complete_analysis("path/to/data.csv")
```

### Custom Analysis
```python
# Load data only
analysis.load_data("path/to/data.csv")

# Perform analysis
analysis.perform_analysis()

# Create specific figures
analysis.create_figure_1_scatter()
analysis.create_figure_2_quadrant()

# Save data
analysis.save_data_files()
```

## 📄 License

This project is developed for the BCCS AI Workshop analysis.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the generated reports
3. Examine the data files for insights

---

**Generated**: August 2025
**Data Source**: BCCS AI Workshop August 11, 2025  
**Analysis Method**: Concept Mapping with MDS and Hierarchical Clustering
