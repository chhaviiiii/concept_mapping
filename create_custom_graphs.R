#!/usr/bin/env Rscript

# Custom Graph Creation Script for BCCS AI Workshop July 27, 2025
# This script allows you to create various custom visualizations from your concept mapping data

library(dplyr)
library(readr)
library(ggplot2)
library(ggrepel)
library(viridis)
library(RColorBrewer)
library(gridExtra)
library(tidyr)

# Load the data
data_dir <- "data/rcmap_july27_2025"
output_dir <- "Figures/custom_graphs"

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Load data
cat("Loading concept mapping data...\n")
statements <- read_csv(file.path(data_dir, "Statements.csv"))
ratings <- read_csv(file.path(data_dir, "Ratings.csv"))
cluster_summary <- read_csv("Figures/july27_2025_analysis/cluster_summary.csv")
importance_feasibility <- read_csv("Figures/july27_2025_analysis/importance_feasibility_summary.csv")

# Function to create a bubble chart
create_bubble_chart <- function() {
  cat("Creating bubble chart...\n")
  
  # Combine data
  bubble_data <- importance_feasibility %>%
    left_join(cluster_summary, by = c("StatementID", "StatementText")) %>%
    mutate(
      Cluster = as.factor(Cluster),
      # Create a short version of statement text for labels
      ShortText = substr(StatementText, 1, 50)
    )
  
  bubble_plot <- ggplot(bubble_data, aes(x = Importance, y = Feasibility, size = Importance + Feasibility, color = Cluster)) +
    geom_point(alpha = 0.7) +
    geom_text_repel(
      aes(label = StatementID),
      size = 3,
      max.overlaps = 20,
      box.padding = 0.5
    ) +
    scale_size_continuous(range = c(3, 8)) +
    scale_color_viridis_d() +
    labs(
      title = "AI in Cancer Care: Importance vs Feasibility Bubble Chart",
      subtitle = "Bubble size represents combined importance and feasibility",
      x = "Importance Rating (1-5)",
      y = "Feasibility Rating (1-5)",
      size = "Combined Score",
      color = "Cluster"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "bottom"
    )
  
  ggsave(file.path(output_dir, "bubble_chart.png"), bubble_plot, 
         width = 14, height = 10, dpi = 300)
  
  return(bubble_plot)
}

# Function to create a radar/spider chart (simplified version)
create_radar_chart <- function() {
  cat("Creating radar chart...\n")
  
  # Calculate cluster averages
  cluster_avg <- importance_feasibility %>%
    left_join(cluster_summary, by = c("StatementID", "StatementText")) %>%
    group_by(Cluster) %>%
    summarise(
      avg_importance = mean(Importance, na.rm = TRUE),
      avg_feasibility = mean(Feasibility, na.rm = TRUE),
      .groups = 'drop'
    )
  
  # Create a simple radar-like visualization using bar chart
  radar_data <- cluster_avg %>%
    pivot_longer(cols = c(avg_importance, avg_feasibility), 
                names_to = "Metric", values_to = "Value") %>%
    mutate(
      Metric = ifelse(Metric == "avg_importance", "Importance", "Feasibility"),
      Metric = factor(Metric, levels = c("Importance", "Feasibility"))
    )
  
  radar_plot <- ggplot(radar_data, aes(x = Metric, y = Value, fill = as.factor(Cluster))) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_viridis_d() +
    labs(
      title = "Cluster Performance: Average Importance vs Feasibility",
      subtitle = "By cluster",
      x = "Metric",
      y = "Average Rating",
      fill = "Cluster"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "bottom"
    ) +
    ylim(0, 5)
  
  ggsave(file.path(output_dir, "radar_chart.png"), radar_plot, 
         width = 10, height = 8, dpi = 300)
  
  return(radar_plot)
}

# Function to create a heatmap
create_heatmap <- function() {
  cat("Creating heatmap...\n")
  
  # Create a heatmap of statements by cluster and rating
  heatmap_data <- importance_feasibility %>%
    left_join(cluster_summary, by = c("StatementID", "StatementText")) %>%
    mutate(
      Cluster = as.factor(Cluster),
      Combined_Score = (Importance + Feasibility) / 2
    ) %>%
    arrange(Cluster, desc(Combined_Score))
  
  heatmap_plot <- ggplot(heatmap_data, aes(x = Cluster, y = reorder(StatementID, Combined_Score), fill = Combined_Score)) +
    geom_tile() +
    scale_fill_viridis(option = "plasma") +
    labs(
      title = "Statement Performance Heatmap",
      subtitle = "Color intensity represents combined importance and feasibility score",
      x = "Cluster",
      y = "Statement ID",
      fill = "Combined Score"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      axis.text.y = element_text(size = 8),
      legend.position = "bottom"
    )
  
  ggsave(file.path(output_dir, "heatmap.png"), heatmap_plot, 
         width = 12, height = 10, dpi = 300)
  
  return(heatmap_plot)
}

# Function to create a quadrant analysis
create_quadrant_analysis <- function() {
  cat("Creating quadrant analysis...\n")
  
  # Calculate means for quadrant boundaries
  mean_importance <- mean(importance_feasibility$Importance, na.rm = TRUE)
  mean_feasibility <- mean(importance_feasibility$Feasibility, na.rm = TRUE)
  
  quadrant_data <- importance_feasibility %>%
    left_join(cluster_summary, by = c("StatementID", "StatementText")) %>%
    mutate(
      Cluster = as.factor(Cluster),
      Quadrant = case_when(
        Importance >= mean_importance & Feasibility >= mean_feasibility ~ "High Priority, High Feasibility",
        Importance >= mean_importance & Feasibility < mean_feasibility ~ "High Priority, Low Feasibility",
        Importance < mean_importance & Feasibility >= mean_feasibility ~ "Low Priority, High Feasibility",
        TRUE ~ "Low Priority, Low Feasibility"
      ),
      Quadrant = factor(Quadrant, levels = c(
        "High Priority, High Feasibility",
        "High Priority, Low Feasibility", 
        "Low Priority, High Feasibility",
        "Low Priority, Low Feasibility"
      ))
    )
  
  quadrant_plot <- ggplot(quadrant_data, aes(x = Importance, y = Feasibility, color = Quadrant)) +
    geom_point(size = 3, alpha = 0.7) +
    geom_text_repel(
      aes(label = StatementID),
      size = 3,
      max.overlaps = 15
    ) +
    geom_hline(yintercept = mean_feasibility, linetype = "dashed", color = "red", alpha = 0.7) +
    geom_vline(xintercept = mean_importance, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = "Quadrant Analysis: AI in Cancer Care Statements",
      subtitle = paste("Mean Importance:", round(mean_importance, 2), "| Mean Feasibility:", round(mean_feasibility, 2)),
      x = "Importance Rating (1-5)",
      y = "Feasibility Rating (1-5)",
      color = "Quadrant"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "bottom"
    )
  
  ggsave(file.path(output_dir, "quadrant_analysis.png"), quadrant_plot, 
         width = 14, height = 10, dpi = 300)
  
  return(quadrant_plot)
}

# Function to create a cluster comparison bar chart
create_cluster_comparison <- function() {
  cat("Creating cluster comparison chart...\n")
  
  cluster_stats <- importance_feasibility %>%
    left_join(cluster_summary, by = c("StatementID", "StatementText")) %>%
    group_by(Cluster) %>%
    summarise(
      n_statements = n(),
      avg_importance = mean(Importance, na.rm = TRUE),
      avg_feasibility = mean(Feasibility, na.rm = TRUE),
      sd_importance = sd(Importance, na.rm = TRUE),
      sd_feasibility = sd(Feasibility, na.rm = TRUE),
      .groups = 'drop'
    )
  
  # Create comparison plot
  comparison_data <- cluster_stats %>%
    pivot_longer(cols = c(avg_importance, avg_feasibility), 
                names_to = "Metric", values_to = "Value") %>%
    mutate(
      Metric = ifelse(Metric == "avg_importance", "Importance", "Feasibility"),
      Metric = factor(Metric, levels = c("Importance", "Feasibility"))
    )
  
  comparison_plot <- ggplot(comparison_data, aes(x = as.factor(Cluster), y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_brewer(palette = "Set1") +
    labs(
      title = "Cluster Comparison: Average Ratings",
      subtitle = paste("Total statements:", sum(cluster_stats$n_statements)),
      x = "Cluster",
      y = "Average Rating",
      fill = "Metric"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "bottom"
    ) +
    ylim(0, 5)
  
  ggsave(file.path(output_dir, "cluster_comparison.png"), comparison_plot, 
         width = 10, height = 8, dpi = 300)
  
  return(comparison_plot)
}

# Function to create a word cloud (simplified version using statement frequency)
create_statement_frequency_chart <- function() {
  cat("Creating statement frequency chart...\n")
  
  # Create a frequency chart based on combined scores
  freq_data <- importance_feasibility %>%
    left_join(cluster_summary, by = c("StatementID", "StatementText")) %>%
    mutate(
      Cluster = as.factor(Cluster),
      Combined_Score = (Importance + Feasibility) / 2,
      # Create short labels
      ShortLabel = paste0("S", StatementID)
    ) %>%
    arrange(desc(Combined_Score)) %>%
    head(20)  # Top 20 statements
  
  freq_plot <- ggplot(freq_data, aes(x = reorder(ShortLabel, Combined_Score), y = Combined_Score, fill = Cluster)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    scale_fill_viridis_d() +
    coord_flip() +
    labs(
      title = "Top 20 Statements by Combined Score",
      subtitle = "Combined importance and feasibility rating",
      x = "Statement ID",
      y = "Combined Score (Importance + Feasibility) / 2",
      fill = "Cluster"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      legend.position = "bottom"
    )
  
  ggsave(file.path(output_dir, "statement_frequency.png"), freq_plot, 
         width = 12, height = 10, dpi = 300)
  
  return(freq_plot)
}

# Main execution
cat("Creating custom visualizations...\n")

# Create all custom graphs
bubble_chart <- create_bubble_chart()
radar_chart <- create_radar_chart()
heatmap <- create_heatmap()
quadrant_analysis <- create_quadrant_analysis()
cluster_comparison <- create_cluster_comparison()
statement_frequency <- create_statement_frequency_chart()

cat("\n=== Custom Visualizations Complete ===\n")
cat("All graphs saved to:", output_dir, "\n")
cat("Generated graphs:\n")
cat("- bubble_chart.png (Bubble chart of importance vs feasibility)\n")
cat("- radar_chart.png (Cluster performance comparison)\n")
cat("- heatmap.png (Statement performance heatmap)\n")
cat("- quadrant_analysis.png (Quadrant analysis)\n")
cat("- cluster_comparison.png (Cluster comparison bar chart)\n")
cat("- statement_frequency.png (Top statements frequency chart)\n")

cat("\nYou can now view all your custom graphs in the Figures/custom_graphs directory!\n") 