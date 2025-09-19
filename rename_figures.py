#!/usr/bin/env python3
"""
Rename existing figures to match the desired order
"""

import shutil
from pathlib import Path

def rename_figures():
    """Rename figures to match the desired order."""
    figures_dir = Path('concept_mapping_output/figures')
    
    # Create backup directory
    backup_dir = Path('concept_mapping_output/figures_backup')
    backup_dir.mkdir(exist_ok=True)
    
    # Backup existing figures
    for fig_file in figures_dir.glob('*.png'):
        shutil.copy2(fig_file, backup_dir / fig_file.name)
    
    # Rename figures to match desired order
    rename_mapping = {
        'figure_12_point_map.png': 'figure_1_point_map.png',
        'figure_13_cluster_map.png': 'figure_2_cluster_map.png', 
        'figure_14_point_rating_map.png': 'figure_3_point_rating_map.png',
        'figure_15_cluster_rating_map.png': 'figure_4_cluster_rating_map.png',
        'figure_16_pattern_match.png': 'figure_5_pattern_match.png',
        'figure_17_go_zone_plot.png': 'figure_6_go_zone_plot.png'
    }
    
    for old_name, new_name in rename_mapping.items():
        old_path = figures_dir / old_name
        new_path = figures_dir / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"Renamed {old_name} -> {new_name}")
    
    print("Figure renaming completed!")

if __name__ == "__main__":
    rename_figures()
