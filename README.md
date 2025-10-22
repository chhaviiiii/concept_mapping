# PyConceptMap: Open-Source Concept Mapping Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](https://github.com/pyconceptmap/pyconceptmap)

**PyConceptMap** is a comprehensive Python implementation of concept mapping methodology, inspired by [RCMap](https://haimbar.github.io/RCMap/) (Bar & Mentch, 2017). It provides a user-friendly interface for performing concept mapping analysis, including data loading, multidimensional scaling (MDS), clustering, visualization, and report generation.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Programmatic Interface](#programmatic-interface)
- [Generated Outputs](#generated-outputs)
- [Methodology](#methodology)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Complete Concept Mapping Workflow**: From data loading to final reports
- **Multiple Visualization Types**: Point maps, cluster maps, rating maps, go-zone plots, and more
- **Flexible Analysis Options**: Various MDS and clustering methods
- **Comprehensive Reporting**: Detailed statistical analysis and summaries
- **Easy-to-Use Interface**: Both programmatic and command-line interfaces
- **Well-Documented**: Extensive documentation and examples
- **Production-Ready**: Robust error handling and validation

## Installation

### From Source

```bash
git clone https://github.com/pyconceptmap/pyconceptmap.git
cd pyconceptmap
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Scikit-learn >= 1.0.0
- SciPy >= 1.7.0

## Quick Start

### 1. Create Sample Data

```bash
python run_pyconceptmap.py --create_sample_data
```

This creates sample data files in the `./sample_data` folder.

### 2. Run Analysis

```bash
python run_pyconceptmap.py --data_folder ./sample_data
```

### 3. Check Requirements

```bash
python run_pyconceptmap.py --check_requirements
```

## Data Format

PyConceptMap requires four CSV files in your data folder:

### 1. Statements.csv
```csv
StatementID,Statement
1,First statement
2,Second statement
3,Third statement
```

### 2. SortedCards.csv
```csv
SorterID,PileName,Statement_1,Statement_2,Statement_3
1,Pile A,1,2,3
1,Pile B,4,5
2,Pile 1,1,4
2,Pile 2,2,5
```

### 3. Demographics.csv
```csv
RaterID,Age,Experience,Department
1,25,Low,A
2,30,Medium,B
3,35,High,A
```

### 4. Ratings.csv
```csv
RaterID,StatementID,Importance,Feasibility
1,1,4,3
1,2,5,4
2,1,3,2
2,2,4,3
```

## Usage

### Command Line Interface

```bash
# Basic usage
python run_pyconceptmap.py --data_folder /path/to/data

# With custom output folder
python run_pyconceptmap.py --data_folder /path/to/data --output_folder /path/to/output

# With custom parameters
python run_pyconceptmap.py --data_folder /path/to/data --mds_method classical --clustering_method complete

# Create sample data for testing
python run_pyconceptmap.py --create_sample_data
```

### Programmatic Interface

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

# Or run individual steps
analysis.load_data()
analysis.perform_mds()
analysis.perform_clustering()
analysis.analyze_ratings()
analysis.generate_visualizations()
analysis.generate_reports()
```

## Generated Outputs

### Visualizations (8 files)
- **Point Map**: MDS configuration with statement numbers
- **Cluster Map**: Clusters with convex hull boundaries
- **Point Rating Map**: Point sizes based on importance ratings
- **Cluster Rating Map**: Cluster-level rating visualization
- **Pattern Match**: Cluster comparison across dimensions
- **Go-Zone Plot**: Importance vs feasibility quadrant analysis
- **Dendrogram**: Hierarchical clustering tree
- **Parallel Coordinates**: Multi-dimensional cluster comparison

### Reports (7 files)
- **Sorter Summary**: Participant sorting statistics
- **Rater Summary**: Demographics and rating statistics
- **Statement Summary**: Individual statement statistics
- **ANOVA Results**: Between-cluster variance analysis
- **Tukey's HSD**: Pairwise cluster comparisons
- **Cluster Analysis**: Cluster quality and characteristics
- **Comprehensive Report**: Complete analysis summary

## Methodology

PyConceptMap implements the standard concept mapping methodology:

1. **Data Collection**: Participants sort statements into piles and rate them
2. **Co-occurrence Matrix**: Calculate how often statements are grouped together
3. **Multidimensional Scaling**: Create 2D representation of statement relationships
4. **Clustering**: Group similar statements using hierarchical clustering
5. **Rating Analysis**: Analyze importance and feasibility ratings
6. **Visualization**: Create comprehensive plots and maps
7. **Reporting**: Generate detailed statistical reports

### Key Methodological Features

- **Proper Concept Mapping**: Uses sorting data for MDS (not just ratings)
- **Co-occurrence Matrices**: Based on participant grouping behavior
- **Ward's Hierarchical Clustering**: Optimal cluster identification
- **Statistical Analysis**: ANOVA, Tukey's HSD, gap analysis
- **Comprehensive Visualization**: 8 different plot types

## Advanced Features

### Custom Analysis Parameters

```python
# Custom MDS method
analysis.perform_mds(method='classical')

# Custom clustering
analysis.perform_clustering(method='complete', n_clusters=5)

# Custom visualizations
analysis.visualizer.set_color_scheme('viridis')
```

### Data Validation

```python
from pyconceptmap.utils import validate_data

# Validate data consistency
validation_result = validate_data(statements, sorting_data, ratings, demographics)
if not validation_result['valid']:
    print("Data validation errors:", validation_result['errors'])
```

### Custom Visualizations

```python
# Create custom plots
fig = analysis.visualizer.create_point_map(mds_coords, statements)
fig = analysis.visualizer.create_go_zone_plot(statement_summary)
```

## Examples

### Example 1: Basic Analysis

```python
from pyconceptmap import ConceptMappingAnalysis

# Initialize and run analysis
analysis = ConceptMappingAnalysis('./data')
success = analysis.run_complete_analysis()

if success:
    print("Analysis completed successfully!")
    print(f"Results saved to: {analysis.output_folder}")
```

### Example 2: Custom Parameters

```python
# Custom analysis with specific parameters
analysis = ConceptMappingAnalysis('./data', random_state=123)

# Load data
analysis.load_data()

# Custom MDS
analysis.perform_mds(method='classical')

# Custom clustering
analysis.perform_clustering(method='complete', n_clusters=3)

# Generate results
analysis.analyze_ratings()
analysis.generate_visualizations()
analysis.generate_reports()
```

### Example 3: Data Validation

```python
from pyconceptmap.data_handler import DataHandler

# Load and validate data
handler = DataHandler()
statements = handler.load_statements('./data')
sorting_data = handler.load_sorting_data('./data')
ratings = handler.load_ratings('./data')
demographics = handler.load_demographics('./data')

# Check for issues
validation_result = validate_data(statements, sorting_data, ratings, demographics)
print("Validation result:", validation_result)
```

## Troubleshooting

### Common Issues

1. **Missing Files**: Ensure all four CSV files are present in the data folder
2. **Data Format**: Check that CSV files have the correct column names and formats
3. **Dependencies**: Run `python run_pyconceptmap.py --check_requirements` to verify installation
4. **Memory Issues**: For large datasets, consider reducing the number of statements or using sampling

### Getting Help

- Check the [documentation](docs/) for detailed guides
- Review the [examples](examples/) folder for sample code
- Open an [issue](https://github.com/pyconceptmap/pyconceptmap/issues) for bugs or feature requests

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/pyconceptmap/pyconceptmap.git
cd pyconceptmap
pip install -e ".[dev]"
pytest  # Run tests
```

## Citation

If you use PyConceptMap in your research, please cite:

```bibtex
@software{pyconceptmap2024,
  title={PyConceptMap: Open-Source Concept Mapping Tool},
  author={PyConceptMap Development Team},
  year={2024},
  url={https://github.com/pyconceptmap/pyconceptmap}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [RCMap](https://haimbar.github.io/RCMap/) (Bar & Mentch, 2017)
- Built on the concept mapping methodology of Trochim & Kane (2002)
- Thanks to the open-source community for the excellent Python libraries

## Changelog

### Version 0.1.0 (2024-01-01)
- Initial release
- Complete concept mapping workflow
- Multiple visualization types
- Comprehensive reporting
- Command-line and programmatic interfaces
- Full compatibility with RCMap methodology

---

**PyConceptMap** - Making concept mapping accessible in Python!