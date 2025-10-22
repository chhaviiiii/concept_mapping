#!/usr/bin/env python3
"""
PyConceptMap: Main runner script for concept mapping analysis.

This script provides a command-line interface for running concept mapping
analysis using the PyConceptMap package.

Usage:
    python run_pyconceptmap.py --data_folder /path/to/data --output_folder /path/to/output
    python run_pyconceptmap.py --data_folder /path/to/data  # Uses default output folder
    python run_pyconceptmap.py --create_sample_data  # Creates sample data for testing
"""

import argparse
import sys
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from pyconceptmap import ConceptMappingAnalysis, check_requirements, create_sample_data
except ImportError:
    # Try importing from the local directory
    try:
        sys.path.insert(0, '.')
        from pyconceptmap import ConceptMappingAnalysis, check_requirements, create_sample_data
    except ImportError:
        print("PyConceptMap package not found. Please install it first.")
        print("   pip install -e .")
        sys.exit(1)


def main():
    """Main function for the PyConceptMap runner."""
    parser = argparse.ArgumentParser(
        description='PyConceptMap: Open-Source Concept Mapping Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pyconceptmap.py --data_folder ./data
  python run_pyconceptmap.py --data_folder ./data --output_folder ./results
  python run_pyconceptmap.py --create_sample_data
  python run_pyconceptmap.py --check_requirements
        """
    )
    
    parser.add_argument('--data_folder', type=str,
                       help='Path to the data folder containing CSV files')
    parser.add_argument('--output_folder', type=str,
                       help='Path to the output folder (default: data_folder/output)')
    parser.add_argument('--create_sample_data', action='store_true',
                       help='Create sample data files for testing')
    parser.add_argument('--check_requirements', action='store_true',
                       help='Check if all required packages are installed')
    parser.add_argument('--mds_method', type=str, default='smacof',
                       choices=['smacof', 'classical'],
                       help='MDS method to use (default: smacof)')
    parser.add_argument('--clustering_method', type=str, default='ward',
                       choices=['ward', 'complete', 'average', 'single'],
                       help='Clustering method to use (default: ward)')
    parser.add_argument('--auto_select_clusters', action='store_true', default=True,
                       help='Automatically select optimal number of clusters')
    parser.add_argument('--n_clusters', type=int,
                       help='Number of clusters (overrides auto selection)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("PYCONCEPTMAP: OPEN-SOURCE CONCEPT MAPPING TOOL")
    print("=" * 60)
    print("Version: 0.1.0")
    print("Inspired by RCMap (Bar & Mentch, 2017)")
    print("=" * 60)
    
    # Check requirements
    if args.check_requirements:
        print("\nChecking requirements...")
        requirements = check_requirements()
        all_installed = all(requirements.values())
        
        for package, installed in requirements.items():
            status = "✅" if installed else "❌"
            print(f"  {status} {package}")
        
        if not all_installed:
            print("\n❌ Some required packages are missing.")
            print("Please install them using: pip install -r requirements.txt")
            return 1
        else:
            print("\n✅ All requirements satisfied!")
            return 0
    
    # Create sample data
    if args.create_sample_data:
        print("\nCreating sample data...")
        sample_folder = Path('./sample_data')
        success = create_sample_data(sample_folder)
        
        if success:
            print(f"✅ Sample data created in {sample_folder}")
            print("\nSample data structure:")
            print("  - Statements.csv: 10 sample statements")
            print("  - SortedCards.csv: 3 sorters with 3 piles each")
            print("  - Demographics.csv: 3 raters with demographics")
            print("  - Ratings.csv: Importance and feasibility ratings")
            print(f"\nYou can now run: python run_pyconceptmap.py --data_folder {sample_folder}")
            return 0
        else:
            print("❌ Failed to create sample data")
            return 1
    
    # Validate arguments
    if not args.data_folder:
        print("❌ Error: --data_folder is required")
        print("Use --help for usage information")
        return 1
    
    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        print(f"❌ Error: Data folder {data_folder} does not exist")
        return 1
    
    # Set output folder
    if args.output_folder:
        output_folder = Path(args.output_folder)
    else:
        output_folder = data_folder / 'output'
    
    print(f"\nData folder: {data_folder}")
    print(f"Output folder: {output_folder}")
    
    # Initialize analysis
    try:
        analysis = ConceptMappingAnalysis(
            data_folder=data_folder,
            output_folder=output_folder,
            random_state=args.random_state
        )
        
        # Run complete analysis
        print("\nStarting concept mapping analysis...")
        success = analysis.run_complete_analysis(
            mds_method=args.mds_method,
            clustering_method=args.clustering_method,
            auto_select_clusters=args.auto_select_clusters
        )
        
        if success:
            print("\n" + "=" * 60)
            print("CONCEPT MAPPING ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Results saved to: {output_folder}")
            print("\nGenerated files:")
            print("  Visualizations:")
            print("    - point_map.png")
            print("    - cluster_map.png") 
            print("    - point_rating_map.png")
            print("    - cluster_rating_map.png")
            print("    - pattern_match.png")
            print("    - go_zone_plot.png")
            print("    - dendrogram.png")
            print("    - parallel_coordinates.png")
            print("  Reports:")
            print("    - sorter_summary.txt")
            print("    - rater_summary.txt")
            print("    - statement_summary.txt")
            print("    - anova_results.txt")
            print("    - tukey_results.txt")
            print("    - cluster_analysis.txt")
            print("    - comprehensive_report.txt")
            print("  📈 Data:")
            print("    - StatementSummaryXX.csv")
            return 0
        else:
            print("\n❌ Concept mapping analysis failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
