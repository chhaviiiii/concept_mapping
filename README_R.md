# R Implementation - Concept Mapping Analysis

Complete R implementation of concept mapping analysis with data transformation, multidimensional scaling, clustering, and advanced visualizations.

## 🚀 Quick Start

### 1. Install Dependencies
```r
# Install required packages
install.packages(c(
  "dplyr", "readr", "ggplot2", "ggrepel", "cluster", "factoextra", 
  "MASS", "corrplot", "viridis", "RColorBrewer", "gridExtra", 
  "tidyr", "stringr", "purrr", "tibble", "knitr"
), repos = "https://cran.rstudio.com/")
```

### 2. Run Complete Analysis
```bash
Rscript run_analysis.R
```

### 3. Or Run Individual Steps
```bash
# Transform data
Rscript transform_data_to_rcmap.R

# Run analysis
Rscript concept_mapping_analysis.R

# Create custom visualizations
Rscript create_custom_graphs.R

# Generate HTML report
Rscript create_html_report.R
```

## 📁 R Project Structure

```
├── R/                                    # R implementation files
│   ├── transform_data_to_rcmap.R         # Data transformation
│   ├── concept_mapping_analysis.R         # Main analysis
│   ├── create_custom_graphs.R             # Custom visualizations
│   ├── create_html_report.R               # HTML report generator
│   └── run_analysis.R                     # Master script
├── data/rcmap_analysis/                   # R-formatted data
│   ├── Statements.csv                     # 100 statements
│   ├── Ratings.csv                        # Participant ratings
│   ├── Demographics.csv                   # Participant info
│   └── SortedCards.csv                    # Grouping data
├── Figures/analysis/                      # Generated visualizations
│   ├── concept_map.png                    # MDS with clusters
│   ├── importance_vs_feasibility.png      # Scatter plot
│   ├── rating_distribution.png            # Histograms
│   ├── cluster_analysis.png               # WSS and silhouette
│   ├── similarity_heatmap.png             # Correlation matrix
│   ├── summary_statistics.csv             # Key metrics
│   └── statements_with_clusters.csv       # Results
└── requirements.R                         # R dependencies
```

## 🔧 Requirements

### Core Packages
- **dplyr** (≥1.0.0) - Data manipulation
- **readr** (≥2.0.0) - Fast data reading
- **ggplot2** (≥3.3.0) - Grammar of graphics
- **ggrepel** (≥0.9.0) - Label positioning
- **cluster** (≥2.1.0) - Clustering algorithms
- **factoextra** (≥1.0.7) - Extract and visualize results
- **MASS** (≥7.3.0) - Multidimensional scaling
- **corrplot** (≥0.92) - Correlation matrix visualization

### Optional Packages
- **viridis** (≥0.6.0) - Color palettes
- **RColorBrewer** (≥1.1.0) - Color schemes
- **gridExtra** (≥2.3) - Arrange multiple plots
- **knitr** (≥1.40) - Report generation

## 📊 Analysis Features

### 1. Data Transformation
- Converts Qualtrics survey data to RCMap format
- Handles CSV and TSV files with UTF-8 encoding
- Extracts statements, ratings, and demographics
- Creates structured datasets for analysis

### 2. Multidimensional Scaling (MDS)
- Converts rating patterns to 2D coordinates
- Uses correlation-based similarity matrix
- Handles missing data and edge cases
- Provides foundation for clustering

### 3. Clustering Analysis
- **K-means clustering** with optimal k selection
- **Elbow method** for WSS analysis
- **Silhouette analysis** for cluster quality
- Automatic selection of best number of clusters

### 4. Visualizations
- **Concept Map**: MDS plot with color-coded clusters
- **Importance vs Feasibility**: Scatter plot with mean lines
- **Rating Distribution**: Histograms for both rating types
- **Cluster Analysis**: WSS and silhouette plots
- **Similarity Heatmap**: Correlation matrix visualization

### 5. Statistical Analysis
- Correlation between importance and feasibility
- Mean ratings by statement and cluster
- Cluster quality metrics
- Summary statistics and reporting

## 🎯 Key Advantages of R Version

### Statistical Rigor
- **Comprehensive statistical testing** capabilities
- **Publication-quality graphics** with ggplot2
- **Advanced clustering methods** available
- **Robust data validation** and cleaning

### Visualization Quality
- **Professional appearance** suitable for publications
- **Consistent styling** across all plots
- **Customizable themes** and color schemes
- **High-resolution output** for presentations

### Research Integration
- **Seamless integration** with other R packages
- **Reproducible research** with R Markdown
- **Statistical reporting** standards
- **Academic workflow** compatibility

## 📈 Comparison: R vs Python

| Feature | R Version | Python Version |
|---------|-----------|----------------|
| **Data Transformation** | ✅ | ✅ |
| **MDS Analysis** | ✅ | ✅ |
| **Clustering** | ✅ | ✅ |
| **Basic Visualizations** | ✅ | ✅ |
| **Statistical Testing** | **Excellent** | Good |
| **Publication Graphics** | **Excellent** | Good |
| **Research Integration** | **Excellent** | Good |
| **Performance** | Good | **Better** |
| **Interactive Plots** | Limited | **Excellent** |
| **Learning Curve** | Steep | **Gentler** |

## 🔍 Usage Examples

### Basic Analysis
```r
# Load the analysis script
source("simplified_concept_mapping_analysis.R")

# The script will automatically:
# 1. Load and validate data
# 2. Perform MDS analysis
# 3. Find optimal clusters
# 4. Create visualizations
# 5. Generate summary statistics
```

### Custom Analysis
```r
# Load data manually
statements <- read_csv("data/rcmap_july27_2025/Statements.csv")
ratings <- read_csv("data/rcmap_july27_2025/Ratings.csv")

# Custom MDS
rating_matrix <- create_rating_matrix(ratings)
similarity_matrix <- cor(t(rating_matrix), use = "pairwise.complete.obs")
distance_matrix <- 1 - abs(similarity_matrix)
mds_result <- cmdscale(distance_matrix, k = 2)

# Custom clustering
optimal_k <- find_optimal_clusters(mds_result)
cluster_result <- kmeans(mds_result, centers = optimal_k)
```

### Custom Visualizations
```r
# Create custom concept map
ggplot(data.frame(
  x = mds_result[,1], 
  y = mds_result[,2],
  cluster = factor(cluster_result$cluster),
  statement = statements$StatementText
)) +
  geom_point(aes(x = x, y = y, color = cluster), size = 3) +
  geom_text_repel(aes(x = x, y = y, label = statement), size = 2) +
  theme_minimal() +
  labs(title = "Custom Concept Map")
```

## 🚀 Advanced Features

### 1. HTML Report Generation
```r
# Generate comprehensive HTML report
source("create_html_report.R")
```

### 2. Custom Visualizations
```r
# Create additional custom plots
source("create_custom_graphs.R")
```

### 3. Statistical Testing
```r
# ANOVA for cluster differences
cluster_groups <- split(ratings$Rating, cluster_result$cluster)
anova_result <- aov(Rating ~ cluster, data = ratings)
summary(anova_result)
```

### 4. Publication Graphics
```r
# Set publication theme
theme_publication <- function() {
  theme_minimal() +
    theme(
      text = element_text(size = 12, family = "Times"),
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      legend.position = "bottom"
    )
}
```

## 📝 Troubleshooting

### Common Issues

1. **Package Installation**
   ```r
   # If install.packages fails, try:
   install.packages("package_name", dependencies = TRUE)
   
   # Or use BiocManager for bioinformatics packages
   if (!require("BiocManager", quietly = TRUE))
     install.packages("BiocManager")
   BiocManager::install("package_name")
   ```

2. **Memory Issues**
   ```r
   # Clear memory
   gc()
   
   # Increase memory limit (Windows)
   memory.limit(size = 8000)
   ```

3. **Encoding Issues**
   ```r
   # Set encoding for file reading
   read_csv("file.csv", locale = locale(encoding = "UTF-8"))
   ```

### Performance Optimization
```r
# Use data.table for large datasets
library(data.table)
ratings_dt <- fread("data/rcmap_july27_2025/Ratings.csv")

# Use parallel processing
library(parallel)
library(doParallel)
```

## 🔮 Future Enhancements

### Planned Features
- **Interactive Shiny dashboard** for real-time analysis
- **Advanced clustering algorithms** (hierarchical, DBSCAN)
- **Network analysis** for statement relationships
- **Automated report generation** with R Markdown
- **API integration** capabilities
- **Database connectivity** options

### Custom Extensions
- **Mixed-methods analysis** integration
- **Longitudinal concept mapping** support
- **Multi-group comparisons** and testing
- **Geospatial analysis** for location-based insights

## 📄 License

This R implementation is provided as-is for educational and research purposes. The code is designed to be reusable for any concept mapping analysis study.

## 🤝 Contributing

To extend this analysis:
1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue in the repository 