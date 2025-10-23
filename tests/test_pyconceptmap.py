#!/usr/bin/env python3
"""
Test script for PyConceptMap to ensure everything works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        from pyconceptmap import ConceptMappingAnalysis, DataHandler, ConceptMapVisualizer, ReportGenerator
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from pyconceptmap.utils import validate_data, check_requirements, create_sample_data
        print("✅ Utility imports successful")
    except ImportError as e:
        print(f"❌ Utility import error: {e}")
        return False
    
    return True

def test_requirements():
    """Test that all requirements are met."""
    print("\nTesting requirements...")
    
    try:
        from pyconceptmap.utils import check_requirements
        requirements = check_requirements()
        
        all_good = True
        for package, installed in requirements.items():
            if installed:
                print(f"✅ {package}")
            else:
                print(f"❌ {package} - NOT INSTALLED")
                all_good = False
        
        return all_good
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def test_sample_data():
    """Test creating sample data."""
    print("\nTesting sample data creation...")
    
    try:
        from pyconceptmap.utils import create_sample_data
        sample_folder = Path('./test_sample_data')
        
        success = create_sample_data(sample_folder)
        if success:
            print("✅ Sample data created successfully")
            
            # Check that files exist
            required_files = ['Statements.csv', 'SortedCards.csv', 'Demographics.csv', 'Ratings.csv']
            for file_name in required_files:
                if (sample_folder / file_name).exists():
                    print(f"✅ {file_name} exists")
                else:
                    print(f"❌ {file_name} missing")
                    return False
            
            return True
        else:
            print("❌ Sample data creation failed")
            return False
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def test_analysis():
    """Test running a complete analysis."""
    print("\nTesting complete analysis...")
    
    try:
        from pyconceptmap import ConceptMappingAnalysis
        
        # Use the sample data we just created
        sample_folder = Path('./test_sample_data')
        output_folder = Path('./test_output')
        
        # Initialize analysis
        analysis = ConceptMappingAnalysis(
            data_folder=sample_folder,
            output_folder=output_folder,
            random_state=42
        )
        
        print("✅ Analysis initialized")
        
        # Run complete analysis
        success = analysis.run_complete_analysis()
        
        if success:
            print("✅ Complete analysis successful")
            
            # Check that output files were created
            expected_files = [
                'point_map.png',
                'cluster_map.png',
                'point_rating_map.png',
                'cluster_rating_map.png',
                'pattern_match.png',
                'go_zone_plot.png',
                'dendrogram.png',
                'parallel_coordinates.png'
            ]
            
            for file_name in expected_files:
                if (output_folder / file_name).exists():
                    print(f"✅ {file_name} created")
                else:
                    print(f"❌ {file_name} missing")
                    return False
            
            return True
        else:
            print("❌ Complete analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PYCONCEPTMAP TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Requirements Test", test_requirements),
        ("Sample Data Test", test_sample_data),
        ("Analysis Test", test_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\nALL TESTS PASSED! PyConceptMap is working correctly.")
        return 0
    else:
        print(f"\n{len(results) - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
