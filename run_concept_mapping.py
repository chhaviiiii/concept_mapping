#!/usr/bin/env python3
"""
Concept Mapping Analysis Runner
Executes comprehensive concept mapping analysis for BCCS AI Workshop data
"""

from concept_mapping_analysis import ConceptMappingAnalysis
import sys
from pathlib import Path

def main():
    """Run the concept mapping analysis."""
    print("🎯 BCCS AI WORKSHOP - CONCEPT MAPPING ANALYSIS")
    print("=" * 60)
    
    # Check if data file exists
    data_path = Path("data/BCCS AI Workshop_August 11, 2025_23.45.csv")
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists in the data/ directory.")
        return
    
    try:
        # Initialize and run analysis
        analysis = ConceptMappingAnalysis()
        analysis.run_complete_analysis(str(data_path))
        
        print("\n🎉 Analysis completed successfully!")
        print("📁 Check the 'concept_mapping_output' directory for results.")
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
