import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
from datetime import datetime

class August11ReportGenerator:
    def __init__(self):
        """Initialize the report generator."""
        self.data_dir = "data/rcmap_august11_2025"
        self.figures_dir = "Figures/august11_2025_analysis"
        self.custom_dir = "Figures/custom_graphs_august11"
        self.output_file = "BCCS_AI_Workshop_August11_2025_Report.pdf"
        
        # Load data
        self.statements_df = pd.read_csv(f'{self.data_dir}/Statements.csv')
        self.ratings_df = pd.read_csv(f'{self.data_dir}/Ratings.csv')
        self.importance_feasibility_df = pd.read_csv(f'{self.figures_dir}/importance_feasibility_summary.csv')
        self.cluster_ratings_df = pd.read_csv(f'{self.figures_dir}/cluster_ratings.csv')
        
        # Load July 27 statements to get actual statement text
        self.july27_statements_df = pd.read_csv('data/rcmap_july27_2025/Statements.csv')
        
        # Get top statements
        self.get_top_statements()
        
        print("Report generator initialized")
    
    def get_top_statements(self):
        """Get the most important and feasible statements."""
        # Most important statements
        self.top_important = self.importance_feasibility_df.nlargest(5, 'Importance_Mean')
        
        # Most feasible statements
        self.top_feasible = self.importance_feasibility_df.nlargest(5, 'Feasibility_Mean')
        
        # Get actual statement text from July 27 data
        july27_with_actual_text = self.july27_statements_df.copy()
        july27_with_actual_text = july27_with_actual_text.rename(columns={'StatementText': 'ActualStatementText'})
        
        self.top_important = self.top_important.merge(july27_with_actual_text, on='StatementID')
        self.top_feasible = self.top_feasible.merge(july27_with_actual_text, on='StatementID')
    
    def create_title_page(self, story):
        """Create the title page."""
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=getSampleStyleSheet()['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        title = Paragraph("BCCS AI Workshop 2025:<br/>Concept Mapping Analysis Report", title_style)
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
        
        subtitle = Paragraph("August 11, 2025 Dataset Analysis", subtitle_style)
        story.append(subtitle)
        story.append(Spacer(1, 0.3*inch))
        
        # Date
        date_style = ParagraphStyle(
            'Date',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.gray
        )
        
        date_text = Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}", date_style)
        story.append(date_text)
        story.append(Spacer(1, 0.5*inch))
        
        # Summary stats
        stats_style = ParagraphStyle(
            'Stats',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.black
        )
        
        total_statements = len(self.statements_df)
        total_participants = len(self.ratings_df['ParticipantID'].unique())
        total_ratings = len(self.ratings_df)
        
        stats_text = f"""
        <b>Dataset Summary:</b><br/>
        • Total Statements Analyzed: {total_statements}<br/>
        • Total Participants: {total_participants}<br/>
        • Total Ratings Collected: {total_ratings}<br/>
        • Analysis Date: August 11, 2025
        """
        
        stats = Paragraph(stats_text, stats_style)
        story.append(stats)
        story.append(PageBreak())
    
    def create_table_of_contents(self, story):
        """Create table of contents."""
        toc_style = ParagraphStyle(
            'TOC',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
        
        toc = Paragraph("Table of Contents", toc_style)
        story.append(toc)
        story.append(Spacer(1, 0.2*inch))
        
        # TOC items
        toc_items = [
            "1. Overview",
            "2. Key Findings",
            "3. Visualizations",
            "4. Conclusion"
        ]
        
        toc_style_item = ParagraphStyle(
            'TOCItem',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=12,
            spaceAfter=8,
            leftIndent=20
        )
        
        for item in toc_items:
            toc_item = Paragraph(item, toc_style_item)
            story.append(toc_item)
        
        story.append(PageBreak())
    
    def create_overview(self, story):
        """Create the overview section."""
        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.darkblue
        )
        
        overview_header = Paragraph("1. Overview", header_style)
        story.append(overview_header)
        
        # Overview content
        content_style = ParagraphStyle(
            'Content',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        overview_text = """
        This report presents the comprehensive concept mapping analysis conducted for the BCCS AI Workshop held on August 11, 2025. 
        The analysis involved 100 AI-related statements that were evaluated by 15 participants across two key dimensions: 
        <b>importance</b> and <b>feasibility</b>.
        
        The concept mapping methodology combines qualitative grouping of statements with quantitative rating scales to identify 
        patterns, priorities, and strategic insights. Participants first grouped similar statements together, then rated each 
        statement on a 1-5 scale for both importance and feasibility.
        
        The analysis reveals four distinct clusters of statements, each representing different strategic priorities and 
        implementation considerations for AI integration in cancer care. The visualizations provide insights into the 
        relationships between importance and feasibility, helping to identify immediate action items, strategic planning 
        priorities, and areas requiring further consideration.
        """
        
        overview = Paragraph(overview_text, content_style)
        story.append(overview)
        story.append(PageBreak())
    
    def create_key_findings(self, story):
        """Create the key findings section."""
        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.darkblue
        )
        
        findings_header = Paragraph("2. Key Findings", header_style)
        story.append(findings_header)
        
        # Content style
        content_style = ParagraphStyle(
            'Content',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        # Most Important Statements
        important_header = Paragraph("<b>Most Important Statements:</b>", content_style)
        story.append(important_header)
        story.append(Spacer(1, 0.1*inch))
        
        for idx, row in self.top_important.iterrows():
            importance_text = f"• <b>Statement {row['StatementID']}</b>: {row['ActualStatementText']} (Importance: {row['Importance_Mean']:.2f})"
            important_item = Paragraph(importance_text, content_style)
            story.append(important_item)
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Most Feasible Statements
        feasible_header = Paragraph("<b>Most Feasible Statements:</b>", content_style)
        story.append(feasible_header)
        story.append(Spacer(1, 0.1*inch))
        
        for idx, row in self.top_feasible.iterrows():
            feasibility_text = f"• <b>Statement {row['StatementID']}</b>: {row['ActualStatementText']} (Feasibility: {row['Feasibility_Mean']:.2f})"
            feasible_item = Paragraph(feasibility_text, content_style)
            story.append(feasible_item)
            story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Cluster Summary
        cluster_header = Paragraph("<b>Cluster Analysis Summary:</b>", content_style)
        story.append(cluster_header)
        story.append(Spacer(1, 0.1*inch))
        
        cluster_summary = f"""
        The analysis identified four distinct clusters of statements:
        
        • <b>Cluster 0</b> (23 statements): High importance (3.73) and medium feasibility (3.42) - Strategic priorities requiring planning
        • <b>Cluster 1</b> (15 statements): Low importance (2.64) and low feasibility (2.53) - Monitor and evaluate
        • <b>Cluster 2</b> (45 statements): Medium importance (3.14) and medium feasibility (3.07) - Balanced considerations
        • <b>Cluster 3</b> (17 statements): High importance (3.59) and high feasibility (4.00) - Immediate action items
        """
        
        cluster_text = Paragraph(cluster_summary, content_style)
        story.append(cluster_text)
        story.append(PageBreak())
    
    def create_visualizations(self, story):
        """Create the visualizations section."""
        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.darkblue
        )
        
        viz_header = Paragraph("3. Visualizations", header_style)
        story.append(viz_header)
        
        # Content style
        content_style = ParagraphStyle(
            'Content',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        # List of visualizations to include
        visualizations = [
            {
                'file': f'{self.figures_dir}/importance_vs_feasibility.png',
                'title': 'Figure 1: Importance vs Feasibility Scatter Plot',
                'description': 'This scatter plot shows the relationship between importance and feasibility ratings for all 100 statements. The red dashed lines represent the median values, dividing the plot into four quadrants. Statements in the upper-right quadrant (high importance, high feasibility) represent immediate action items, while those in the upper-left (high importance, low feasibility) require strategic planning.'
            },
            {
                'file': f'{self.custom_dir}/quadrant_analysis.png',
                'title': 'Figure 2: Quadrant Analysis',
                'description': 'Enhanced quadrant analysis showing the strategic positioning of statements. The size of each point represents the importance rating, while colors indicate the quadrant. This visualization helps identify priority areas for implementation and strategic planning.'
            },
            {
                'file': f'{self.custom_dir}/bubble_chart.png',
                'title': 'Figure 3: Bubble Chart Analysis',
                'description': 'Bubble chart where bubble size represents importance and color represents the gap between importance and feasibility. Red colors indicate statements where importance exceeds feasibility (strategic priorities), while blue colors indicate statements where feasibility exceeds importance (quick wins).'
            },
            {
                'file': f'{self.figures_dir}/optimal_clusters_analysis.png',
                'title': 'Figure 4: Optimal Clusters Analysis',
                'description': 'Comprehensive cluster analysis showing the elbow method, silhouette analysis, and gap statistic for determining the optimal number of clusters. The analysis confirms that 4 clusters provide the best balance of interpretability and statistical validity.'
            },
            {
                'file': f'{self.custom_dir}/cluster_comparison.png',
                'title': 'Figure 5: Cluster Comparison',
                'description': 'Visualization of the four clusters identified in the analysis. The left panel shows the scatter plot with cluster assignments, while the right panel displays the mean importance and feasibility ratings for each cluster, helping to understand the characteristics of each group.'
            },
            {
                'file': f'{self.custom_dir}/similarity_heatmap.png',
                'title': 'Figure 6: Statement Similarity Heatmap',
                'description': 'Heatmap showing the similarity between statements based on their importance and feasibility ratings. Darker colors indicate higher similarity, revealing patterns and relationships between different statements in the dataset.'
            },
            {
                'file': f'{self.custom_dir}/radar_chart.png',
                'title': 'Figure 7: Top 5 Most Important Statements - Radar Chart',
                'description': 'Radar chart comparing importance and feasibility ratings for the top 5 most important statements. This visualization helps identify which high-priority statements also have high feasibility for immediate implementation.'
            },
            {
                'file': f'{self.custom_dir}/heatmap.png',
                'title': 'Figure 8: Importance and Feasibility Heatmaps',
                'description': 'Heatmaps showing the distribution of importance and feasibility ratings across participants and statements. The left panel shows importance ratings, while the right panel shows feasibility ratings, revealing patterns in participant responses.'
            },
            {
                'file': f'{self.custom_dir}/statement_frequency.png',
                'title': 'Figure 9: Statement Grouping Frequency Analysis',
                'description': 'Analysis of how frequently statements were grouped together by participants. The top panel shows the most frequently grouped statements, while the bottom panel shows the distribution of grouping frequencies, indicating which concepts were most commonly associated.'
            },
            {
                'file': f'{self.figures_dir}/gap_analysis.png',
                'title': 'Figure 10: Gap Analysis',
                'description': 'Bar chart showing the gap between importance and feasibility ratings for each statement. Positive gaps indicate statements where importance exceeds feasibility (strategic priorities), while negative gaps indicate statements where feasibility exceeds importance (quick wins).'
            }
        ]
        
        # Add each visualization
        for i, viz in enumerate(visualizations):
            if os.path.exists(viz['file']):
                # Figure title
                title_style = ParagraphStyle(
                    'FigureTitle',
                    parent=getSampleStyleSheet()['Heading2'],
                    fontSize=12,
                    spaceAfter=10,
                    textColor=colors.darkblue
                )
                
                title = Paragraph(viz['title'], title_style)
                story.append(title)
                
                # Add image
                img = Image(viz['file'], width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.1*inch))
                
                # Add description
                description = Paragraph(viz['description'], content_style)
                story.append(description)
                story.append(Spacer(1, 0.2*inch))
                
                # Add page break after every 2 figures
                if (i + 1) % 2 == 0 and i < len(visualizations) - 1:
                    story.append(PageBreak())
            else:
                print(f"Warning: Figure file not found: {viz['file']}")
        
        story.append(PageBreak())
    
    def create_conclusion(self, story):
        """Create the conclusion section."""
        # Section header
        header_style = ParagraphStyle(
            'SectionHeader',
            parent=getSampleStyleSheet()['Heading1'],
            fontSize=16,
            spaceAfter=15,
            textColor=colors.darkblue
        )
        
        conclusion_header = Paragraph("4. Conclusion", header_style)
        story.append(conclusion_header)
        
        # Content style
        content_style = ParagraphStyle(
            'Content',
            parent=getSampleStyleSheet()['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        conclusion_text = """
        The concept mapping analysis of the August 11, 2025 BCCS AI Workshop reveals clear patterns and priorities for AI integration in cancer care. The analysis of 100 statements by 15 participants provides valuable insights into both the importance and feasibility of various AI applications.
        
        <b>Key Strategic Insights:</b>
        
        • <b>Immediate Action Items (Cluster 3)</b>: 17 statements with high importance and high feasibility represent opportunities for immediate implementation. These should be prioritized for quick wins and demonstration projects.
        
        • <b>Strategic Planning Priorities (Cluster 0)</b>: 23 statements with high importance but medium feasibility require careful planning and resource allocation. These represent the core strategic initiatives for AI integration.
        
        • <b>Balanced Considerations (Cluster 2)</b>: 45 statements with medium importance and feasibility represent the majority of AI applications that require ongoing evaluation and incremental implementation.
        
        • <b>Monitor and Evaluate (Cluster 1)</b>: 15 statements with low importance and feasibility should be monitored for future consideration as technology and priorities evolve.
        
        The analysis demonstrates that participants recognize both the potential benefits and challenges of AI integration in cancer care. The clear identification of high-priority, high-feasibility items provides a roadmap for immediate action, while the strategic planning items require careful consideration of resources, timelines, and stakeholder engagement.
        
        This concept mapping exercise serves as a foundation for developing a comprehensive AI strategy in cancer care, balancing immediate opportunities with long-term strategic planning.
        """
        
        conclusion = Paragraph(conclusion_text, content_style)
        story.append(conclusion)
    
    def generate_report(self):
        """Generate the complete PDF report."""
        print(f"Generating PDF report: {self.output_file}")
        
        # Create the PDF document
        doc = SimpleDocTemplate(self.output_file, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        # Build the story
        story = []
        
        # Add sections
        self.create_title_page(story)
        self.create_table_of_contents(story)
        self.create_overview(story)
        self.create_key_findings(story)
        self.create_visualizations(story)
        self.create_conclusion(story)
        
        # Build the PDF
        doc.build(story)
        
        print(f"PDF report generated successfully: {self.output_file}")
        print(f"Report includes {len(story)} elements across multiple pages")

if __name__ == "__main__":
    generator = August11ReportGenerator()
    generator.generate_report() 