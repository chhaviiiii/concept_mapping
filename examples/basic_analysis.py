#!/usr/bin/env python3
"""
Basic PyConceptMap Analysis Example

This example demonstrates how to perform a complete concept mapping analysis
using PyConceptMap with sample data.
"""

from pyconceptmap import ConceptMappingAnalysis
from pathlib import Path

def main():
    """Run a basic concept mapping analysis."""
    print("=" * 60)
    print("PYCONCEPTMAP: BASIC ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("Step 1: Creating sample data...")
    from pyconceptmap.utils import create_sample_data
    
    data_folder = Path('./example_data')
    create_sample_data(data_folder)
    print(f"✅ Sample data created in {data_folder}")
    
    # Step 2: Initialize analysis
    print("\nStep 2: Initializing analysis...")
    analysis = ConceptMappingAnalysis(
        data_folder=data_folder,
        output_folder=Path('./example_output'),
        random_state=42
    )
    print("✅ Analysis initialized")
    
    # Step 3: Run complete analysis
    print("\nStep 3: Running complete analysis...")
    success = analysis.run_complete_analysis()
    
    if success:
        print("✅ Analysis completed successfully!")
        print(f"Results saved to: {analysis.output_folder}")
        
        # Step 4: Show results summary
        print("\nStep 4: Analysis Results Summary")
        print("-" * 40)
        print(f"Number of statements: {len(analysis.statements)}")
        print(f"Number of sorters: {len(analysis.sorting_data)}")
        print(f"Number of clusters: {analysis.n_clusters}")
        print(f"Cluster sizes: {analysis.cluster_labels}")
        
        # Step 5: List generated files
        print("\nStep 5: Generated Files")
        print("-" * 40)
        output_files = list(analysis.output_folder.glob('*'))
        for file_path in sorted(output_files):
            if file_path.is_file():
                print(f"📄 {file_path.name}")
        
        print("\n🎉 Basic analysis example completed successfully!")
        
    else:
        print("❌ Analysis failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
