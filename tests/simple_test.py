#!/usr/bin/env python3
"""
Simple test to verify PyConceptMap works.
"""

import sys
from pathlib import Path

def main():
    print("Testing PyConceptMap...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from pyconceptmap import ConceptMappingAnalysis
        print("✅ Imports successful")
        
        # Test sample data creation
        print("2. Creating sample data...")
        from pyconceptmap.utils import create_sample_data
        sample_folder = Path('./simple_test_data')
        success = create_sample_data(sample_folder)
        
        if not success:
            print("❌ Sample data creation failed")
            return 1
        
        print("✅ Sample data created")
        
        # Test analysis
        print("3. Running analysis...")
        analysis = ConceptMappingAnalysis(
            data_folder=sample_folder,
            output_folder=Path('./simple_test_output'),
            random_state=42
        )
        
        # Load data
        if not analysis.load_data():
            print("❌ Data loading failed")
            return 1
        
        print("✅ Data loaded")
        
        # Run MDS
        if not analysis.perform_mds():
            print("❌ MDS failed")
            return 1
        
        print("✅ MDS completed")
        
        # Run clustering
        if not analysis.perform_clustering():
            print("❌ Clustering failed")
            return 1
        
        print("✅ Clustering completed")
        
        # Run rating analysis
        if not analysis.analyze_ratings():
            print("❌ Rating analysis failed")
            return 1
        
        print("✅ Rating analysis completed")
        
        # Generate visualizations
        if not analysis.generate_visualizations():
            print("❌ Visualization generation failed")
            return 1
        
        print("✅ Visualizations generated")
        
        # Generate reports
        if not analysis.generate_reports():
            print("❌ Report generation failed")
            return 1
        
        print("✅ Reports generated")
        
        print("\nALL TESTS PASSED! PyConceptMap is working correctly.")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
