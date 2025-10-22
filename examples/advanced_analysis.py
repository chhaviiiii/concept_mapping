#!/usr/bin/env python3
"""
Advanced PyConceptMap Analysis Example

This example demonstrates advanced features including custom parameters,
step-by-step analysis, and custom visualizations.
"""

from pyconceptmap import ConceptMappingAnalysis
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    """Run an advanced concept mapping analysis."""
    print("=" * 60)
    print("PYCONCEPTMAP: ADVANCED ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("Step 1: Creating sample data...")
    from pyconceptmap.utils import create_sample_data
    
    data_folder = Path('./advanced_example_data')
    create_sample_data(data_folder)
    print(f"✅ Sample data created in {data_folder}")
    
    # Step 2: Initialize analysis with custom parameters
    print("\nStep 2: Initializing analysis with custom parameters...")
    analysis = ConceptMappingAnalysis(
        data_folder=data_folder,
        output_folder=Path('./advanced_example_output'),
        random_state=123  # Custom random seed
    )
    print("✅ Analysis initialized")
    
    # Step 3: Step-by-step analysis
    print("\nStep 3: Step-by-step analysis...")
    
    # Load data
    print("  - Loading data...")
    if not analysis.load_data():
        print("❌ Data loading failed")
        return 1
    print("  ✅ Data loaded successfully")
    
    # Perform MDS with different methods
    print("  - Performing MDS (classical method)...")
    if not analysis.perform_mds(method='classical'):
        print("❌ MDS failed")
        return 1
    print("  ✅ MDS completed")
    
    # Perform clustering with custom parameters
    print("  - Performing clustering (complete linkage)...")
    if not analysis.perform_clustering(method='complete', n_clusters=3):
        print("❌ Clustering failed")
        return 1
    print("  ✅ Clustering completed")
    
    # Analyze ratings
    print("  - Analyzing ratings...")
    if not analysis.analyze_ratings():
        print("❌ Rating analysis failed")
        return 1
    print("  ✅ Rating analysis completed")
    
    # Step 4: Custom visualizations
    print("\nStep 4: Creating custom visualizations...")
    
    # Set custom color scheme
    analysis.visualizer.set_color_scheme('viridis')
    
    # Generate visualizations
    if not analysis.generate_visualizations():
        print("❌ Visualization generation failed")
        return 1
    print("  ✅ Custom visualizations created")
    
    # Step 5: Custom reports
    print("\nStep 5: Generating custom reports...")
    if not analysis.generate_reports():
        print("❌ Report generation failed")
        return 1
    print("  ✅ Custom reports generated")
    
    # Step 6: Access analysis results
    print("\nStep 6: Analysis Results")
    print("-" * 40)
    print(f"Number of statements: {len(analysis.statements)}")
    print(f"Number of sorters: {len(analysis.sorting_data)}")
    print(f"Number of clusters: {analysis.n_clusters}")
    print(f"Cluster sizes: {analysis.cluster_labels}")
    
    # Show cluster statistics
    if analysis.cluster_means is not None:
        print("\nCluster Statistics:")
        print(analysis.cluster_means)
    
    # Show statement summary
    if analysis.statement_summary is not None:
        print(f"\nStatement Summary Shape: {analysis.statement_summary.shape}")
        print("First few rows:")
        print(analysis.statement_summary.head())
    
    # Step 7: Custom analysis
    print("\nStep 7: Performing custom analysis...")
    
    # Calculate additional statistics
    if analysis.statement_summary is not None:
        importance_col = 'Importance_mean'
        feasibility_col = 'Feasibility_mean'
        
        if importance_col in analysis.statement_summary.columns:
            avg_importance = analysis.statement_summary[importance_col].mean()
            print(f"Average Importance Rating: {avg_importance:.3f}")
        
        if feasibility_col in analysis.statement_summary.columns:
            avg_feasibility = analysis.statement_summary[feasibility_col].mean()
            print(f"Average Feasibility Rating: {avg_feasibility:.3f}")
    
    # Step 8: List generated files
    print("\nStep 8: Generated Files")
    print("-" * 40)
    output_files = list(analysis.output_folder.glob('*'))
    for file_path in sorted(output_files):
        if file_path.is_file():
            print(f"{file_path.name}")
    
    print("\nAdvanced analysis example completed successfully!")
    print(f"Results saved to: {analysis.output_folder}")
    
    return 0

if __name__ == '__main__':
    exit(main())
