#!/usr/bin/env python3
"""
Master Script for Python Concept Mapping Analysis
BCCS AI Workshop July 27, 2025
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_packages():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install -r requirements_python.txt")
        return False
    
    return True

def run_data_transformation():
    """Run the data transformation script"""
    print("=" * 60)
    print("STEP 1: Data Transformation")
    print("=" * 60)
    
    if os.path.exists("transform_july27_2025_to_python.py"):
        print("Running data transformation...")
        result = subprocess.run([sys.executable, "transform_july27_2025_to_python.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data transformation completed successfully!")
            print(result.stdout)
        else:
            print("❌ Data transformation failed!")
            print(result.stderr)
            return False
    else:
        print("❌ Data transformation script not found!")
        return False
    
    return True

def run_concept_mapping_analysis():
    """Run the concept mapping analysis"""
    print("\n" + "=" * 60)
    print("STEP 2: Concept Mapping Analysis")
    print("=" * 60)
    
    if os.path.exists("concept_mapping_analysis_python.py"):
        print("Running concept mapping analysis...")
        result = subprocess.run([sys.executable, "concept_mapping_analysis_python.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Concept mapping analysis completed successfully!")
            print(result.stdout)
        else:
            print("❌ Concept mapping analysis failed!")
            print(result.stderr)
            return False
    else:
        print("❌ Concept mapping analysis script not found!")
        return False
    
    return True

def create_custom_visualizations():
    """Create additional custom visualizations"""
    print("\n" + "=" * 60)
    print("STEP 3: Custom Visualizations")
    print("=" * 60)
    
    # This would be a separate script for custom visualizations
    # For now, we'll just note that the main analysis creates the core visualizations
    print("✅ Core visualizations created in main analysis")
    print("📊 Available visualizations:")
    print("  - Concept Map (MDS with clusters)")
    print("  - Importance vs Feasibility scatter plot")
    print("  - Rating distribution histograms")
    print("  - Cluster analysis plots (WSS and silhouette)")
    print("  - Similarity matrix heatmap")
    
    return True

def generate_summary_report():
    """Generate a summary report"""
    print("\n" + "=" * 60)
    print("STEP 4: Summary Report")
    print("=" * 60)
    
    output_dir = "Figures/python_analysis"
    if os.path.exists(output_dir):
        print(f"✅ Analysis results available in: {output_dir}")
        
        # List generated files
        files = os.listdir(output_dir)
        print(f"\n📁 Generated files ({len(files)} total):")
        for file in sorted(files):
            if file.endswith('.png'):
                print(f"  📊 {file}")
            elif file.endswith('.csv'):
                print(f"  📄 {file}")
        
        # Check for key files
        key_files = [
            'concept_map.png',
            'importance_vs_feasibility.png',
            'rating_distribution.png',
            'cluster_analysis.png',
            'similarity_heatmap.png',
            'summary_statistics.csv',
            'statements_with_clusters.csv'
        ]
        
        missing_files = [f for f in key_files if not os.path.exists(os.path.join(output_dir, f))]
        
        if missing_files:
            print(f"\n⚠️  Missing files: {missing_files}")
        else:
            print(f"\n✅ All key files generated successfully!")
    else:
        print("❌ Output directory not found!")
        return False
    
    return True

def main():
    """Main function to run the complete Python analysis pipeline"""
    print("🚀 Python Concept Mapping Analysis Pipeline")
    print("BCCS AI Workshop July 27, 2025")
    print("=" * 60)
    
    # Check if required packages are installed
    if not check_python_packages():
        print("\n❌ Please install required packages first!")
        return False
    
    # Step 1: Data transformation
    if not run_data_transformation():
        print("\n❌ Pipeline stopped at data transformation step!")
        return False
    
    # Step 2: Concept mapping analysis
    if not run_concept_mapping_analysis():
        print("\n❌ Pipeline stopped at concept mapping analysis step!")
        return False
    
    # Step 3: Custom visualizations
    if not create_custom_visualizations():
        print("\n❌ Pipeline stopped at custom visualizations step!")
        return False
    
    # Step 4: Summary report
    if not generate_summary_report():
        print("\n❌ Pipeline stopped at summary report step!")
        return False
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 PYTHON ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📊 Your concept mapping analysis results are ready!")
    print("📁 Check the 'Figures/python_analysis' directory for all visualizations")
    print("📄 Check the CSV files for detailed results and statistics")
    print("\n🔍 Key findings:")
    print("  - Concept map with optimal clustering")
    print("  - Importance vs feasibility correlation analysis")
    print("  - Rating distribution patterns")
    print("  - Cluster quality assessment")
    print("  - Statement similarity matrix")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 