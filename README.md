# Concept Mapping Analysis for BCCS AI Workshop July 27, 2025

A comprehensive concept mapping analysis system for AI applications in cancer care, featuring data transformation, analysis, and visualization capabilities.

## 🚀 Quick Start

1. **Transform your data:**
   ```bash
   Rscript transform_july27_2025_to_rcmap.R
   ```

2. **Run the analysis:**
   ```bash
   Rscript run_july27_2025_analysis.R
   ```

3. **Generate custom visualizations:**
   ```bash
   Rscript create_custom_graphs.R
   ```

4. **Create HTML report:**
   ```bash
   Rscript create_html_report.R
   ```

## 📁 Project Structure

```
├── data/                           # Raw data files
│   ├── BCCS AI Workshop_July 27, 2025_15.23.csv
│   └── BCCS AI Workshop_July 27, 2025_15.26_utf8.tsv
├── data/rcmap_july27_2025/         # Transformed RCMap format data
├── Figures/                        # Generated visualizations
│   ├── july27_2025_analysis/       # Main analysis plots
│   └── custom_graphs/              # Additional custom visualizations
├── transform_july27_2025_to_rcmap.R    # Data transformation script
├── simplified_concept_mapping_analysis.R # Main analysis script
├── create_custom_graphs.R          # Custom visualization generator
├── create_html_report.R            # HTML report generator
├── run_july27_2025_analysis.R      # Master script to run everything
└── README_July27_2025_Analysis.md  # Detailed documentation
```

## 📊 Key Features

- **Data Transformation**: Converts Qualtrics survey data to RCMap format
- **Concept Mapping Analysis**: Multidimensional scaling and clustering
- **Rating Analysis**: Importance vs feasibility correlation analysis
- **Custom Visualizations**: Bubble charts, heatmaps, quadrant analysis
- **Professional Reports**: HTML reports with captions and recommendations

## 🎯 Results

The analysis identified:
- **3 distinct conceptual clusters** of AI applications in cancer care
- **Moderate positive correlation** (r = 0.51) between importance and feasibility
- **Strategic quadrants** for implementation planning
- **Top 20 priority statements** for immediate action

## 📈 Visualizations Generated

1. **Concept Map** - Multidimensional scaling with clusters
2. **Quadrant Analysis** - Strategic priority framework
3. **Bubble Chart** - Comprehensive overview with cluster info
4. **Importance vs Feasibility** - Core relationship analysis
5. **Heatmap** - Statement performance visualization
6. **Cluster Comparison** - Group performance analysis
7. **Top Statements** - Priority recommendations
8. **Radar Chart** - Cluster performance overview

## 🔧 Requirements

- R (version 4.0+)
- Required packages: dplyr, readr, stringr, tidyr, data.table, purrr, ggplot2, ggrepel, cluster, factoextra, MASS, corrplot, viridis, RColorBrewer, gridExtra, knitr, tibble

## 📝 Usage

See `README_July27_2025_Analysis.md` for detailed usage instructions and troubleshooting.

## 📄 License

This project is part of the BCCS AI Workshop analysis conducted on July 27, 2025.
