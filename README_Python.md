# Python Concept Mapping Analysis
## BCCS AI Workshop July 27, 2025

A complete Python implementation of concept mapping analysis for AI applications in cancer care, featuring data transformation, multidimensional scaling, clustering, and advanced visualizations.

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements_python.txt
```

### 2. Run Complete Analysis
```bash
python run_python_analysis.py
```

### 3. Or Run Individual Steps
```bash
# Transform data
python transform_july27_2025_to_python.py

# Run analysis
python concept_mapping_analysis_python.py
```

## 📁 Python Project Structure

```
├── Python/                               # Python implementation files
│   ├── transform_data_to_python.py       # Data transformation
│   ├── concept_mapping_analysis_python.py # Main analysis
│   └── run_python_analysis.py            # Master script
├── data/python_analysis/                 # Python-formatted data
│   ├── statements.csv                    # 100 statements
│   ├── ratings.csv                       # Participant ratings
│   ├── demographics.csv                  # Participant info
│   └── sorted_cards.csv                  # Grouping data
├── Figures/python_analysis/              # Generated visualizations
│   ├── concept_map.png                   # MDS with clusters
│   ├── importance_vs_feasibility.png     # Scatter plot
│   ├── rating_distribution.png           # Histograms
│   ├── cluster_analysis.png              # WSS and silhouette
│   ├── similarity_heatmap.png            # Correlation matrix
│   ├── summary_statistics.csv            # Key metrics
│   └── statements_with_clusters.csv      # Results
└── requirements_python.txt               # Python dependencies
```

## 🔧 Requirements

### Core Packages
- **pandas** (≥1.5.0) - Data manipulation
- **numpy** (≥1.21.0) - Numerical computing
- **matplotlib** (≥3.5.0) - Basic plotting
- **seaborn** (≥0.11.0) - Statistical visualizations
- **scikit-learn** (≥1.1.0) - Machine learning
- **scipy** (≥1.9.0) - Scientific computing

### Optional Packages
- **plotly** (≥5.0.0) - Interactive plots
- **jupyter** (≥1.0.0) - Jupyter notebooks
- **dash** (≥2.0.0) - Interactive dashboards

## 📊 Analysis Features

### 1. Data Transformation
- Converts Qualtrics survey data to Python format
- Handles CSV and TSV files
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

## 🎯 Key Advantages of Python Version

### Performance
- **Faster execution** with optimized libraries
- **Memory efficient** data handling
- **Parallel processing** capabilities

### Flexibility
- **Object-oriented design** for easy extension
- **Modular architecture** for custom analyses
- **Rich ecosystem** of additional libraries

### Visualization
- **Interactive plots** with Plotly
- **Publication-quality** graphics with Matplotlib/Seaborn
- **Customizable styling** and themes

### Integration
- **Jupyter notebooks** for interactive analysis
- **Web dashboards** with Dash/Streamlit
- **API integration** capabilities
- **Database connectivity** options

## 📈 Comparison: R vs Python

| Feature | R Version | Python Version |
|---------|-----------|----------------|
| **Data Transformation** | ✅ | ✅ |
| **MDS Analysis** | ✅ | ✅ |
| **Clustering** | ✅ | ✅ |
| **Basic Visualizations** | ✅ | ✅ |
| **Performance** | Good | **Better** |
| **Interactive Plots** | Limited | **Excellent** |
| **Machine Learning** | Good | **Excellent** |
| **Web Integration** | Limited | **Excellent** |
| **Package Ecosystem** | Good | **Excellent** |
| **Learning Curve** | Steep | **Gentler** |

## 🔍 Usage Examples

### Basic Analysis
```python
from concept_mapping_analysis_python import ConceptMappingAnalysis

# Initialize analysis
analysis = ConceptMappingAnalysis()

# Run complete analysis
results = analysis.run_analysis()

# Access results
statements = results['statements']
mds_coords = results['mds_coords']
cluster_labels = results['cluster_labels']
```

### Custom Analysis
```python
# Load data manually
analysis = ConceptMappingAnalysis()
analysis.load_data()

# Custom MDS
rating_matrix = analysis.prepare_rating_matrix()
mds_coords, similarity_matrix = analysis.perform_mds(rating_matrix)

# Custom clustering
optimal_k, wss, silhouette_scores, k_range = analysis.find_optimal_clusters(mds_coords)
cluster_labels, kmeans = analysis.perform_clustering(mds_coords, optimal_k)
```

### Interactive Visualization
```python
import plotly.express as px
import plotly.graph_objects as go

# Create interactive concept map
fig = px.scatter(
    x=mds_coords[:, 0], 
    y=mds_coords[:, 1],
    color=cluster_labels,
    hover_data={'Statement': statements['StatementText']},
    title="Interactive Concept Map"
)
fig.show()
```

## 🚀 Advanced Features

### 1. Interactive Dashboards
```python
# Create Dash dashboard (requires dash package)
import dash
from dash import dcc, html
import plotly.graph_objs as go

app = dash.Dash(__name__)
# Add dashboard components...
```

### 2. Machine Learning Extensions
```python
# Add advanced clustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

# Add dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

### 3. Statistical Testing
```python
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ANOVA for cluster differences
f_stat, p_value = stats.f_oneway(*cluster_groups)
```

## 📝 Troubleshooting

### Common Issues

1. **Package Installation**
   ```bash
   # If pip fails, try conda
   conda install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

2. **Memory Issues**
   ```python
   # Reduce memory usage
   import gc
   gc.collect()
   ```

3. **Plot Display**
   ```python
   # For Jupyter notebooks
   %matplotlib inline
   
   # For headless servers
   import matplotlib
   matplotlib.use('Agg')
   ```

### Performance Optimization
```python
# Use efficient data types
ratings_df = ratings_df.astype({
    'StatementID': 'int8',
    'Rating': 'int8'
})

# Use parallel processing
from joblib import Parallel, delayed
```

## 🔮 Future Enhancements

### Planned Features
- **Interactive web dashboard** with Dash
- **Advanced clustering algorithms** (DBSCAN, HDBSCAN)
- **Natural language processing** for statement analysis
- **Real-time data processing** capabilities
- **API endpoints** for integration
- **Automated report generation** with Jinja2

### Custom Extensions
- **Deep learning** for pattern recognition
- **Network analysis** for statement relationships
- **Time series analysis** for longitudinal studies
- **Geospatial analysis** for location-based insights

## 📄 License

This Python implementation is provided as-is for educational and research purposes. The code is designed to be reusable for any concept mapping analysis study.

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