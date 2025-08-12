import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
from datetime import datetime

class StrategicPrioritiesAnalyzer:
    def __init__(self):
        """Initialize the strategic priorities analyzer."""
        self.data_dir = "data/rcmap_august11_2025"
        self.figures_dir = "Figures/august11_2025_analysis"
        self.output_file = "Strategic_Priorities_August11_2025.pdf"
        
        # Load data
        self.importance_feasibility_df = pd.read_csv(f'{self.figures_dir}/importance_feasibility_summary.csv')
        self.july27_statements_df = pd.read_csv('data/rcmap_july27_2025/Statements.csv')
        
        # Get actual statement text
        self.get_statement_text()
        
        # Calculate quadrants
        self.calculate_quadrants()
        
        print("Strategic priorities analyzer initialized")
    
    def get_statement_text(self):
        """Get actual statement text from July 27 data."""
        july27_with_actual_text = self.july27_statements_df.copy()
        july27_with_actual_text = july27_with_actual_text.rename(columns={'StatementText': 'ActualStatementText'})
        
        self.importance_feasibility_df = self.importance_feasibility_df.merge(july27_with_actual_text, on='StatementID')
    
    def calculate_quadrants(self):
        """Calculate which quadrant each statement belongs to."""
        # Calculate medians for quadrant boundaries
        importance_median = self.importance_feasibility_df['Importance_Mean'].median()
        feasibility_median = self.importance_feasibility_df['Feasibility_Mean'].median()
        
        # Assign quadrants
        self.importance_feasibility_df['Quadrant'] = 'Other'
        
        # High Priority, High Feasibility (Immediate Implementation)
        mask_high_high = (self.importance_feasibility_df['Importance_Mean'] >= importance_median) & \
                        (self.importance_feasibility_df['Feasibility_Mean'] >= feasibility_median)
        self.importance_feasibility_df.loc[mask_high_high, 'Quadrant'] = 'Immediate Implementation'
        
        # High Priority, Low Feasibility (Research and Development)
        mask_high_low = (self.importance_feasibility_df['Importance_Mean'] >= importance_median) & \
                       (self.importance_feasibility_df['Feasibility_Mean'] < feasibility_median)
        self.importance_feasibility_df.loc[mask_high_low, 'Quadrant'] = 'Research and Development'
        
        # Low Priority, High Feasibility (Quick Wins)
        mask_low_high = (self.importance_feasibility_df['Importance_Mean'] < importance_median) & \
                       (self.importance_feasibility_df['Feasibility_Mean'] >= feasibility_median)
        self.importance_feasibility_df.loc[mask_low_high, 'Quadrant'] = 'Quick Wins'
        
        # Low Priority, Low Feasibility (Monitor)
        mask_low_low = (self.importance_feasibility_df['Importance_Mean'] < importance_median) & \
                      (self.importance_feasibility_df['Feasibility_Mean'] < feasibility_median)
        self.importance_feasibility_df.loc[mask_low_low, 'Quadrant'] = 'Monitor'
    
    def get_top_statements_by_quadrant(self):
        """Get top 3 statements for each strategic quadrant."""
        quadrants = ['Immediate Implementation', 'Research and Development', 'Quick Wins']
        top_statements = {}
        
        for quadrant in quadrants:
            quadrant_data = self.importance_feasibility_df[self.importance_feasibility_df['Quadrant'] == quadrant]
            
            if quadrant == 'Immediate Implementation':
                # Sort by combined score (importance + feasibility)
                quadrant_data['Combined_Score'] = quadrant_data['Importance_Mean'] + quadrant_data['Feasibility_Mean']
                top_3 = quadrant_data.nlargest(3, 'Combined_Score')
            elif quadrant == 'Research and Development':
                # Sort by importance (highest priority)
                top_3 = quadrant_data.nlargest(3, 'Importance_Mean')
            else:  # Quick Wins
                # Sort by feasibility (easiest to implement)
                top_3 = quadrant_data.nlargest(3, 'Feasibility_Mean')
            
            top_statements[quadrant] = top_3
        
        return top_statements
    
    def create_quadrant_visualization(self):
        """Create a focused quadrant visualization."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate medians
        importance_median = self.importance_feasibility_df['Importance_Mean'].median()
        feasibility_median = self.importance_feasibility_df['Feasibility_Mean'].median()
        
        # Define colors for quadrants
        colors_map = {
            'Immediate Implementation': '#2ca02c',  # Green
            'Research and Development': '#d62728',  # Red
            'Quick Wins': '#1f77b4',  # Blue
            'Monitor': '#7f7f7f'  # Gray
        }
        
        # Plot all points
        for quadrant in colors_map.keys():
            quadrant_data = self.importance_feasibility_df[self.importance_feasibility_df['Quadrant'] == quadrant]
            ax.scatter(quadrant_data['Feasibility_Mean'], quadrant_data['Importance_Mean'],
                      c=colors_map[quadrant], label=quadrant, alpha=0.6, s=50)
        
        # Highlight top 3 statements for each strategic quadrant
        top_statements = self.get_top_statements_by_quadrant()
        
        for quadrant, statements in top_statements.items():
            if quadrant in ['Immediate Implementation', 'Research and Development', 'Quick Wins']:
                ax.scatter(statements['Feasibility_Mean'], statements['Importance_Mean'],
                          c=colors_map[quadrant], s=200, edgecolors='black', linewidth=2, alpha=0.8)
                
                # Add statement numbers
                for idx, row in statements.iterrows():
                    ax.annotate(f"{row['StatementID']}", 
                               (row['Feasibility_Mean'], row['Importance_Mean']),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, fontweight='bold')
        
        # Add quadrant lines
        ax.axhline(y=importance_median, color='black', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=feasibility_median, color='black', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add quadrant labels
        ax.text(0.95, 0.95, 'Immediate\nImplementation', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#2ca02c', alpha=0.8, edgecolor='black'))
        ax.text(0.05, 0.95, 'Research &\nDevelopment', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#d62728', alpha=0.8, edgecolor='black'))
        ax.text(0.95, 0.05, 'Quick Wins', 
                transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.8, edgecolor='black'))
        
        # Customize plot
        ax.set_xlabel('Feasibility Rating (Mean)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Importance Rating (Mean)', fontsize=14, fontweight='bold')
        ax.set_title('Strategic Priorities - August 11 BCCS AI Workshop\n(Top 3 statements highlighted for each strategic quadrant)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')
        
        plt.tight_layout()
        plt.savefig('strategic_priorities_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Created strategic priorities visualization")
    
    def create_strategic_report(self):
        """Create a focused strategic priorities report."""
        print(f"Generating strategic priorities report: {self.output_file}")
        
        # Create the PDF document
        doc = SimpleDocTemplate(self.output_file, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        # Build the story
        story = []
        
        # Title page
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=getSampleStyleSheet()['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        title = Paragraph("Strategic Priorities Analysis<br/>BCCS AI Workshop - August 11, 2025", title_style)
        story.append(title)
        story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.gray
        )
        
        subtitle = Paragraph("Top 3 Statements for Each Strategic Category", subtitle_style)
        story.append(subtitle)
        story.append(PageBreak())
        
        # Get top statements
        top_statements = self.get_top_statements_by_quadrant()
        
        # Content style
        content_style = ParagraphStyle(
            'Content',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        # Header style
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.darkblue
        )
        
        # 1. Immediate Implementation
        immediate_header = Paragraph("1. Immediate Implementation (High Priority, High Feasibility)", header_style)
        story.append(immediate_header)
        
        immediate_desc = Paragraph("These statements represent the highest priority items that are also highly feasible to implement. They should be prioritized for immediate action and can serve as quick wins to demonstrate value.", content_style)
        story.append(immediate_desc)
        story.append(Spacer(1, 0.1*inch))
        
        for idx, row in top_statements['Immediate Implementation'].iterrows():
            combined_score = row['Importance_Mean'] + row['Feasibility_Mean']
            statement_text = f"• <b>Statement {row['StatementID']}</b>: {row['ActualStatementText']}<br/>  &nbsp;&nbsp;&nbsp;&nbsp;Importance: {row['Importance_Mean']:.2f} | Feasibility: {row['Feasibility_Mean']:.2f} | Combined Score: {combined_score:.2f}"
            statement_item = Paragraph(statement_text, content_style)
            story.append(statement_item)
            story.append(Spacer(1, 0.05*inch))
        
        story.append(PageBreak())
        
        # 2. Research and Development
        research_header = Paragraph("2. Research and Development (High Priority, Low Feasibility)", header_style)
        story.append(research_header)
        
        research_desc = Paragraph("These statements represent high-priority items that require significant research, development, or strategic planning. They should be included in long-term strategic initiatives.", content_style)
        story.append(research_desc)
        story.append(Spacer(1, 0.1*inch))
        
        for idx, row in top_statements['Research and Development'].iterrows():
            gap = row['Importance_Mean'] - row['Feasibility_Mean']
            statement_text = f"• <b>Statement {row['StatementID']}</b>: {row['ActualStatementText']}<br/>  &nbsp;&nbsp;&nbsp;&nbsp;Importance: {row['Importance_Mean']:.2f} | Feasibility: {row['Feasibility_Mean']:.2f} | Gap: {gap:.2f}"
            statement_item = Paragraph(statement_text, content_style)
            story.append(statement_item)
            story.append(Spacer(1, 0.05*inch))
        
        story.append(PageBreak())
        
        # 3. Quick Wins
        quick_header = Paragraph("3. Quick Wins (Low Priority, High Feasibility)", header_style)
        story.append(quick_header)
        
        quick_desc = Paragraph("These statements represent items that are easy to implement but may not be the highest priority. They can be implemented quickly to build momentum and demonstrate progress.", content_style)
        story.append(quick_desc)
        story.append(Spacer(1, 0.1*inch))
        
        for idx, row in top_statements['Quick Wins'].iterrows():
            gap = row['Feasibility_Mean'] - row['Importance_Mean']
            statement_text = f"• <b>Statement {row['StatementID']}</b>: {row['ActualStatementText']}<br/>  &nbsp;&nbsp;&nbsp;&nbsp;Importance: {row['Importance_Mean']:.2f} | Feasibility: {row['Feasibility_Mean']:.2f} | Gap: {gap:.2f}"
            statement_item = Paragraph(statement_text, content_style)
            story.append(statement_item)
            story.append(Spacer(1, 0.05*inch))
        
        story.append(PageBreak())
        
        # Add visualization
        if os.path.exists('strategic_priorities_visualization.png'):
            viz_header = Paragraph("Strategic Priorities Visualization", header_style)
            story.append(viz_header)
            
            img = Image('strategic_priorities_visualization.png', width=6*inch, height=5*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            
            viz_desc = Paragraph("This visualization shows all statements plotted by importance vs feasibility. The highlighted points represent the top 3 statements for each strategic category. Green points are Immediate Implementation, red points are Research & Development, and blue points are Quick Wins.", content_style)
            story.append(viz_desc)
        
        # Build the PDF
        doc.build(story)
        
        print(f"Strategic priorities report generated successfully: {self.output_file}")
    
    def print_summary(self):
        """Print a summary of the strategic priorities."""
        top_statements = self.get_top_statements_by_quadrant()
        
        print("\n" + "="*80)
        print("STRATEGIC PRIORITIES - AUGUST 11 BCCS AI WORKSHOP")
        print("="*80)
        
        for quadrant, statements in top_statements.items():
            print(f"\n{quadrant.upper()}")
            print("-" * len(quadrant))
            
            for idx, row in statements.iterrows():
                if quadrant == 'Immediate Implementation':
                    combined_score = row['Importance_Mean'] + row['Feasibility_Mean']
                    print(f"• Statement {row['StatementID']}: {row['ActualStatementText']}")
                    print(f"  Importance: {row['Importance_Mean']:.2f} | Feasibility: {row['Feasibility_Mean']:.2f} | Combined: {combined_score:.2f}")
                elif quadrant == 'Research and Development':
                    gap = row['Importance_Mean'] - row['Feasibility_Mean']
                    print(f"• Statement {row['StatementID']}: {row['ActualStatementText']}")
                    print(f"  Importance: {row['Importance_Mean']:.2f} | Feasibility: {row['Feasibility_Mean']:.2f} | Gap: {gap:.2f}")
                else:  # Quick Wins
                    gap = row['Feasibility_Mean'] - row['Importance_Mean']
                    print(f"• Statement {row['StatementID']}: {row['ActualStatementText']}")
                    print(f"  Importance: {row['Importance_Mean']:.2f} | Feasibility: {row['Feasibility_Mean']:.2f} | Gap: {gap:.2f}")
                print()
    
    def run_analysis(self):
        """Run the complete strategic priorities analysis."""
        print("Running strategic priorities analysis...")
        
        # Create visualization
        self.create_quadrant_visualization()
        
        # Create report
        self.create_strategic_report()
        
        # Print summary
        self.print_summary()

if __name__ == "__main__":
    analyzer = StrategicPrioritiesAnalyzer()
    analyzer.run_analysis() 