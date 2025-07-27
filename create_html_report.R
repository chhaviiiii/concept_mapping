#!/usr/bin/env Rscript

# HTML Report Generator for BCCS AI Workshop July 27, 2025 Concept Mapping Analysis
# This script creates a comprehensive HTML report with all visualizations and captions

library(dplyr)
library(readr)

# Load data for the report
data_dir <- "data/rcmap_july27_2025"
cluster_summary <- read_csv("Figures/july27_2025_analysis/cluster_summary.csv")
importance_feasibility <- read_csv("Figures/july27_2025_analysis/importance_feasibility_summary.csv")

# Create HTML content
html_content <- paste0('
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BCCS AI Workshop July 27, 2025: Concept Mapping Analysis Report</title>
    <style>
        body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 40px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 30px;
        }
        .figure {
            margin: 30px 0;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .caption {
            font-style: italic;
            color: #555;
            margin-top: 10px;
            text-align: left;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .key-findings {
            background-color: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }
        .recommendations {
            background-color: #f0f8f0;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #27ae60;
        }
        .toc {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            margin: 8px 0;
        }
        .toc a {
            text-decoration: none;
            color: #3498db;
            font-weight: 500;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .stats-box {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }
        .print-button {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .print-button:hover {
            background-color: #2980b9;
        }
        @media print {
            .print-button {
                display: none;
            }
            body {
                background-color: white;
            }
            .container {
                box-shadow: none;
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <button class="print-button" onclick="window.print()">Print/Save as PDF</button>
    
    <div class="container">
        <h1>BCCS AI Workshop July 27, 2025: Concept Mapping Analysis Report</h1>
        
        <div class="stats-box">
            <strong>Study Overview:</strong> 16 participants rated 100 statements about AI in cancer care on importance and feasibility dimensions. Analysis revealed 3 distinct conceptual clusters with a moderate positive correlation (r = 0.51) between importance and feasibility ratings.
        </div>

        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#main-visualizations">Main Visualizations</a></li>
                <li><a href="#methodology">Methodology</a></li>
                <li><a href="#results">Results</a></li>
                <li><a href="#detailed-analysis">Detailed Analysis</a></li>
                <li><a href="#conclusions">Conclusions and Recommendations</a></li>
            </ul>
        </div>

        <div id="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="key-findings">
                <h3>Key Findings</h3>
                <ul>
                    <li><strong>3 distinct conceptual clusters</strong> were identified through rating pattern analysis</li>
                    <li><strong>Moderate positive correlation</strong> (r = 0.51) between importance and feasibility ratings</li>
                    <li><strong>Top priority areas</strong> include clinician focus enhancement and human oversight</li>
                    <li><strong>High feasibility areas</strong> include documentation efficiency and automation</li>
                </ul>
            </div>
        </div>

        <div id="main-visualizations">
            <h2>Main Visualizations</h2>
            
            <h3>Concept Map Overview</h3>
            <div class="figure">
                <img src="Figures/july27_2025_analysis/concept_map.png" alt="Concept Map">
                <div class="caption">
                    <strong>Figure 1. Concept Map of AI in Cancer Care Statements.</strong> This multidimensional scaling plot shows the 100 statements positioned based on their rating pattern similarity. Statements closer together were rated similarly by participants, while distant statements were rated differently. The color-coded clusters represent three distinct conceptual groups identified through k-means clustering. Statement IDs are labeled for easy reference.
                </div>
            </div>

            <h3>Strategic Priority Analysis</h3>
            <div class="figure">
                <img src="Figures/custom_graphs/quadrant_analysis.png" alt="Quadrant Analysis">
                <div class="caption">
                    <strong>Figure 2. Quadrant Analysis for Strategic Planning.</strong> This plot divides the 100 statements into four strategic quadrants based on mean importance and feasibility scores. The quadrants help identify: (1) High Priority/High Feasibility statements for immediate implementation, (2) High Priority/Low Feasibility statements requiring research and development, (3) Low Priority/High Feasibility statements as quick wins, and (4) Low Priority/Low Feasibility statements for future consideration.
                </div>
            </div>

            <h3>Comprehensive Overview</h3>
            <div class="figure">
                <img src="Figures/custom_graphs/bubble_chart.png" alt="Bubble Chart">
                <div class="caption">
                    <strong>Figure 3. Bubble Chart: Importance vs Feasibility with Cluster Information.</strong> This enhanced scatter plot shows the relationship between importance and feasibility ratings, with bubble size representing the combined importance and feasibility score. Color coding indicates cluster membership, providing a comprehensive view of how statements are positioned across both rating dimensions and conceptual clusters.
                </div>
            </div>
        </div>

        <div id="methodology">
            <h2>Methodology</h2>
            
            <h3>Cluster Analysis Validation</h3>
            <div class="figure">
                <img src="Figures/july27_2025_analysis/wss_plot.png" alt="WSS Plot">
                <div class="caption">
                    <strong>Figure 4. Elbow Plot for Optimal Cluster Determination.</strong> This plot shows the within-cluster sum of squares (WSS) for different numbers of clusters (k=1 to 15). The "elbow" point where the rate of decrease in WSS slows down indicates the optimal number of clusters. In this analysis, the elbow method suggested 3 clusters as optimal for grouping the AI in cancer care statements.
                </div>
            </div>

            <div class="figure">
                <img src="Figures/july27_2025_analysis/silhouette_plot.png" alt="Silhouette Plot">
                <div class="caption">
                    <strong>Figure 5. Silhouette Analysis for Cluster Quality Assessment.</strong> This plot shows the average silhouette width for different numbers of clusters. Higher silhouette values indicate better-defined clusters. The analysis confirms that 3 clusters provide good separation between groups while maintaining reasonable cluster sizes.
                </div>
            </div>

            <div class="figure">
                <img src="Figures/july27_2025_analysis/gap_stat_plot.png" alt="Gap Statistic Plot">
                <div class="caption">
                    <strong>Figure 6. Gap Statistic Plot for Cluster Validation.</strong> This plot compares the within-cluster dispersion of the actual data to that of randomly generated data. The gap statistic helps validate the optimal number of clusters by identifying where the actual data shows the most distinct clustering compared to random data.
                </div>
            </div>

            <h3>Data Distribution</h3>
            <div class="figure">
                <img src="Figures/july27_2025_analysis/rating_distribution.png" alt="Rating Distribution">
                <div class="caption">
                    <strong>Figure 7. Distribution of Mean Ratings by Type.</strong> This histogram shows the frequency distribution of mean importance ratings (blue) and mean feasibility ratings (red) across all 100 statements. The distribution reveals how participants rated the overall importance and feasibility of AI applications in cancer care, with most ratings falling between 3.0 and 4.5 on the 5-point scale.
                </div>
            </div>
        </div>

        <div id="results">
            <h2>Results</h2>
            
            <h3>Core Relationship Analysis</h3>
            <div class="figure">
                <img src="Figures/july27_2025_analysis/importance_vs_feasibility.png" alt="Importance vs Feasibility">
                <div class="caption">
                    <strong>Figure 8. Importance vs Feasibility Scatter Plot.</strong> Each point represents one of the 100 AI in cancer care statements. The x-axis shows importance ratings (1-5 scale) and the y-axis shows feasibility ratings (1-5 scale). Red dashed lines indicate mean values, creating four quadrants: high priority/high feasibility (top right), high priority/low feasibility (top left), low priority/high feasibility (bottom right), and low priority/low feasibility (bottom left).
                </div>
            </div>

            <h3>Cluster Performance Comparison</h3>
            <div class="figure">
                <img src="Figures/custom_graphs/cluster_comparison.png" alt="Cluster Comparison">
                <div class="caption">
                    <strong>Figure 9. Cluster Comparison: Average Ratings by Cluster.</strong> This bar chart compares the average importance and feasibility ratings across the three identified clusters. The chart reveals how different conceptual groups of AI applications in cancer care are perceived in terms of their importance and implementation feasibility.
                </div>
            </div>

            <h3>Detailed Performance Analysis</h3>
            <div class="figure">
                <img src="Figures/custom_graphs/heatmap.png" alt="Heatmap">
                <div class="caption">
                    <strong>Figure 10. Statement Performance Heatmap.</strong> This heatmap visualizes all 100 statements organized by cluster, with color intensity representing the combined importance and feasibility score. Darker colors indicate higher performance scores, making it easy to identify the strongest statements within each cluster and compare performance across clusters.
                </div>
            </div>

            <h3>Top Performing Statements</h3>
            <div class="figure">
                <img src="Figures/custom_graphs/statement_frequency.png" alt="Statement Frequency">
                <div class="caption">
                    <strong>Figure 11. Top 20 Statements by Combined Score.</strong> This horizontal bar chart shows the 20 highest-performing statements based on their combined importance and feasibility scores. The chart helps identify the most promising AI applications for cancer care, with color coding indicating cluster membership.
                </div>
            </div>
        </div>

        <div id="detailed-analysis">
            <h2>Detailed Analysis</h2>
            
            <h3>Cluster Performance Overview</h3>
            <div class="figure">
                <img src="Figures/custom_graphs/radar_chart.png" alt="Radar Chart">
                <div class="caption">
                    <strong>Figure 12. Cluster Performance Radar Chart.</strong> This visualization compares the average importance and feasibility scores for each cluster, providing a quick overview of how the three conceptual groups differ in their perceived value and implementation readiness for AI in cancer care.
                </div>
            </div>
        </div>

        <div id="conclusions">
            <h2>Conclusions and Recommendations</h2>
            
            <div class="key-findings">
                <h3>Key Insights</h3>
                <ol>
                    <li><strong>Three distinct conceptual areas</strong> of AI in cancer care were identified through participant ratings</li>
                    <li><strong>Moderate correlation</strong> between importance and feasibility suggests balanced perceptions</li>
                    <li><strong>High-priority, high-feasibility areas</strong> should be targeted for immediate implementation</li>
                    <li><strong>High-priority, low-feasibility areas</strong> require research and development investment</li>
                </ol>
            </div>

            <div class="recommendations">
                <h3>Strategic Recommendations</h3>
                
                <h4>Immediate Implementation (High Priority, High Feasibility)</h4>
                <ul>
                    <li>Allow clinicians to focus on tasks requiring their expertise</li>
                    <li>Automation of routine tasks</li>
                    <li>Efficiency improvements for documentation</li>
                </ul>

                <h4>Research and Development (High Priority, Low Feasibility)</h4>
                <ul>
                    <li>Human oversight during implementation</li>
                    <li>Concerns with patient acceptability</li>
                    <li>Integration of different information sources</li>
                </ul>

                <h4>Quick Wins (Low Priority, High Feasibility)</h4>
                <ul>
                    <li>Consultation recording and transcription</li>
                    <li>24/7 availability features</li>
                    <li>Digestibility improvements and key point highlighting</li>
                </ul>
            </div>

            <h3>Next Steps</h3>
            <ol>
                <li><strong>Validate findings</strong> with additional stakeholder groups</li>
                <li><strong>Develop implementation roadmaps</strong> for high-priority areas</li>
                <li><strong>Conduct feasibility studies</strong> for low-feasibility, high-importance items</li>
                <li><strong>Monitor progress</strong> on quick-win implementations</li>
            </ol>
        </div>

        <hr style="margin: 40px 0; border: none; border-top: 2px solid #3498db;">
        
        <p style="text-align: center; color: #666; font-style: italic;">
            This report was generated automatically from the concept mapping analysis conducted on July 27, 2025. 
            All visualizations are based on participant ratings of 100 AI in cancer care statements.
        </p>
    </div>
</body>
</html>
')

# Write the HTML file
writeLines(html_content, "concept_mapping_report.html")

cat("HTML report generated successfully!\n")
cat("File: concept_mapping_report.html\n")
cat("You can open this file in any web browser and use the Print button to save as PDF.\n") 