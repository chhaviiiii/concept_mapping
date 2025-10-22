# Installation Guide

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 100MB for installation

## Installation Methods

### Method 1: From Source (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pyconceptmap/pyconceptmap.git
   cd pyconceptmap
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   python run_pyconceptmap.py --check_requirements
   ```

### Method 2: Using pip (Future)

```bash
pip install pyconceptmap
```

## Dependencies

PyConceptMap requires the following Python packages:

- **NumPy** >= 1.21.0 - Numerical computing
- **Pandas** >= 1.3.0 - Data manipulation
- **Matplotlib** >= 3.4.0 - Plotting
- **Seaborn** >= 0.11.0 - Statistical visualization
- **Scikit-learn** >= 1.0.0 - Machine learning algorithms
- **SciPy** >= 1.7.0 - Scientific computing

### Installing Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Verification

After installation, verify that everything works:

```bash
# Check requirements
python run_pyconceptmap.py --check_requirements

# Create sample data
python run_pyconceptmap.py --create_sample_data

# Run test analysis
python run_pyconceptmap.py --data_folder sample_data
```

## Troubleshooting

### Common Installation Issues

1. **Permission Errors**:
   ```bash
   pip install --user -e .
   ```

2. **Missing Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Python Version Issues**:
   ```bash
   python --version  # Should be 3.8+
   ```

4. **Virtual Environment** (Recommended):
   ```bash
   python -m venv pyconceptmap_env
   source pyconceptmap_env/bin/activate  # Linux/Mac
   # or
   pyconceptmap_env\Scripts\activate  # Windows
   pip install -e .
   ```

### Getting Help

If you encounter issues:

1. Check the [troubleshooting section](README.md#troubleshooting)
2. Open an [issue](https://github.com/pyconceptmap/pyconceptmap/issues)
3. Check the [documentation](README.md)

## Development Installation

For developers who want to contribute:

```bash
# Clone and install in development mode
git clone https://github.com/pyconceptmap/pyconceptmap.git
cd pyconceptmap
pip install -e ".[dev]"

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black pyconceptmap/
```
