# Concept Mapping Analysis Toolkit

A comprehensive toolkit for concept mapping analysis with implementations in both **Python** and **R**. This project provides data transformation, multidimensional scaling, clustering analysis, and advanced visualizations for concept mapping studies.

## 🎯 What is Concept Mapping?

Concept mapping is a research methodology that combines qualitative and quantitative approaches to:
- **Generate ideas** through brainstorming sessions
- **Structure concepts** using multidimensional scaling (MDS)
- **Cluster related ideas** using statistical clustering
- **Rate concepts** on importance and feasibility dimensions
- **Visualize relationships** between concepts and clusters

This toolkit is designed for researchers, analysts, and practitioners conducting concept mapping studies in healthcare, education, business, or any domain requiring structured analysis of complex ideas.

## 🚀 Quick Start

### Option 1: Python Implementation (Recommended)
```bash
# Install dependencies
pip install -r requirements_python.txt

# Run complete analysis
python run_python_analysis.py
```

### Option 2: R Implementation
```bash
# Install dependencies
Rscript -e "install.packages(c('dplyr', 'readr', 'ggplot2', 'cluster', 'factoextra', 'MASS', 'corrplot', 'viridis', 'RColorBrewer', 'gridExtra', 'tidyr', 'stringr', 'purrr', 'tibble'), repos='https://cran.rstudio.com/')"

# Run complete analysis
Rscript run_july27_2025_analysis.R
```

## 📁 Project Structure

```
├── README.md                           # This file - main documentation
├── README_Python.md                    # Detailed Python documentation
├── README_R.md                         # Detailed R documentation
├── requirements_python.txt             # Python dependencies
├── requirements.R                      # R dependencies
│
├── data/                               # Raw and processed data
│   ├── BCCS AI Workshop_July 27, 2025_15.23.csv
│   ├── BCCS AI Workshop_July 27, 2025_15.26_utf8.tsv
│   ├── python_july27_2025/            # Python-formatted data
│   └── rcmap_july27_2025/             # R-formatted data
│
├── Python/                             # Python implementation
│   ├── transform_july27_2025_to_python.py
│   ├── concept_mapping_analysis_python.py
│   └── run_python_analysis.py
│
├── R/                                  # R implementation
│   ├── transform_july27_2025_to_rcmap.R
│   ├── simplified_concept_mapping_analysis.R
│   ├── create_custom_graphs.R
│   ├── create_html_report.R
│   └── run_july27_2025_analysis.R
│
└── Figures/                            # Generated visualizations
    ├── python_analysis/               # Python outputs
    └── july27_2025_analysis/          # R outputs
```

## 🔧 Requirements

### Python Version
- **Python 3.8+**
- **Core packages**: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
- **Optional**: plotly, jupyter, dash

### R Version
- **R 4.0+**
- **Core packages**: dplyr, readr, ggplot2, cluster, factoextra, MASS, corrplot
- **Optional**: viridis, RColorBrewer, gridExtra

## 📊 Analysis Features

### Core Analysis
- **Data Transformation**: Convert survey data to analysis format
- **Multidimensional Scaling (MDS)**: Convert rating patterns to 2D coordinates
- **Clustering Analysis**: K-means clustering with optimal k selection
- **Statistical Analysis**: Correlation, descriptive statistics, cluster quality metrics

### Visualizations
- **Concept Maps**: MDS plots with color-coded clusters
- **Importance vs Feasibility**: Scatter plots with strategic quadrants
- **Rating Distributions**: Histograms and box plots
- **Cluster Analysis**: Elbow plots and silhouette analysis
- **Similarity Heatmaps**: Correlation matrix visualizations

### Outputs
- **Interactive plots** (Python)
- **Publication-quality graphics** (both versions)
- **Statistical summaries** in CSV format
- **HTML reports** (R version)

## 🎯 Use Cases

This toolkit is ideal for:
- **Healthcare Research**: Patient care improvement initiatives
- **Educational Planning**: Curriculum development and assessment
- **Business Strategy**: Product development and market analysis
- **Policy Development**: Stakeholder engagement and priority setting
- **Academic Research**: Mixed-methods research studies

## 📈 Example Results

The BCCS AI Workshop analysis (included as example) identified:
- **3 distinct conceptual clusters** of AI applications in cancer care
- **Moderate positive correlation** (r = 0.51) between importance and feasibility
- **Strategic quadrants** for implementation planning
- **Top priority statements** for immediate action

## 🔄 Adapting to Your Data

### Required Data Format
1. **Statements**: List of concepts/ideas to be analyzed
2. **Ratings**: Participant ratings on importance and feasibility scales
3. **Demographics**: Participant information (optional)

### Data Files Structure
```csv
# statements.csv
StatementID,StatementText
1,"Improve patient communication"
2,"Enhance diagnostic accuracy"

# ratings.csv
ParticipantID,StatementID,RatingType,Rating
P1,1,Importance,4
P1,1,Feasibility,3
```

## 🚀 Getting Started with Your Data

### Step 1: Prepare Your Data
- Format your survey data according to the expected structure
- Ensure consistent naming conventions
- Validate data quality and completeness

### Step 2: Choose Your Implementation
- **Python**: Better for machine learning extensions and interactive visualizations
- **R**: Better for statistical analysis and publication-quality graphics

### Step 3: Run the Analysis
- Follow the step-by-step instructions in the respective README files
- Customize parameters as needed for your specific study

### Step 4: Interpret Results
- Review generated visualizations
- Analyze cluster characteristics
- Identify strategic priorities

## 📚 Documentation

- **[Python Documentation](README_Python.md)**: Detailed Python implementation guide
- **[R Documentation](README_R.md)**: Detailed R implementation guide
- **[Example Analysis](README_July27_2025_Analysis.md)**: Complete case study

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📄 License

This project is part of the BCCS AI Workshop analysis conducted on July 27, 2025. The code is provided as-is for educational and research purposes.

## 📞 Support

For questions or issues:
1. Check the troubleshooting sections in the detailed README files
2. Review the code comments for implementation details
3. Create an issue in the repository

## 🙏 Acknowledgments

- **BCCS AI Workshop** participants and organizers
- **RCMap** methodology developers
- **Open source community** for the excellent libraries and tools

---

**Ready to start your concept mapping analysis?** Choose your preferred implementation and follow the detailed guides in the respective README files!
