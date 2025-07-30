#!/usr/bin/env Rscript

# =============================================================================
# Concept Mapping Analysis - R Implementation
# =============================================================================
#
# A comprehensive R implementation of concept mapping analysis featuring:
# - Multidimensional Scaling (MDS) for concept positioning
# - K-means clustering with optimal cluster selection
# - Advanced visualizations and statistical analysis
# - Publication-quality graphics and comprehensive reporting
#
# This implementation is designed for researchers conducting concept mapping studies
# in healthcare, education, business, or any domain requiring structured analysis
# of complex ideas and their relationships.
#
# Author: Concept Mapping Analysis Team
# Date: 2025
# License: Educational and Research Use
# =============================================================================

# Load required libraries for data manipulation, analysis, and visualization
library(dplyr)        # Data manipulation and transformation
library(readr)        # Fast and friendly data reading
library(ggplot2)      # Grammar of graphics for plotting
library(ggrepel)      # Label positioning for plots
library(cluster)      # Clustering algorithms
library(factoextra)   # Extract and visualize clustering results
library(MASS)         # Multidimensional scaling
library(corrplot)     # Correlation matrix visualization
library(viridis)      # Color palettes
library(RColorBrewer) # Color schemes
library(gridExtra)    # Arrange multiple plots
library(tidyr)        # Data tidying
library(stringr)      # String manipulation
library(purrr)        # Functional programming
library(tibble)       # Modern data frames
library(knitr)        # For kable
library(grid)         # For grid.arrange

# =============================================================================
# Configuration and Setup
# =============================================================================

# Set data and output directories
data_dir <- "data/rcmap_analysis"
output_dir <- "Figures/analysis"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("✅ Created output directory:", output_dir, "\n")
}

# Set random seed for reproducible results
set.seed(42)

# =============================================================================
# Data Loading and Validation
# =============================================================================

cat("Loading transformed concept mapping data...\n")

# Load core data files
statements <- read_csv(file.path(data_dir, "Statements.csv"))
ratings <- read_csv(file.path(data_dir, "Ratings.csv"))
demographics <- read_csv(file.path(data_dir, "Demographics.csv"))

# Data validation and quality checks
cat("Validating and cleaning data...\n")

# Check for missing data and provide summary
cat("Data summary:\n")
cat("  - Statements:", nrow(statements), "\n")
cat("  - Participants:", length(unique(ratings$ParticipantID)), "\n")
cat("  - Rating responses:", nrow(ratings), "\n")
cat("  - Rating types:", paste(unique(ratings$RatingType), collapse = ", "), "\n")

# Validate data structure
if (!all(c("StatementID", "StatementText") %in% names(statements))) {
  stop("Statements data missing required columns: StatementID, StatementText")
}

if (!all(c("ParticipantID", "StatementID", "RatingType", "Rating") %in% names(ratings))) {
  stop("Ratings data missing required columns: ParticipantID, StatementID, RatingType, Rating")
}

# Check for missing values
missing_ratings <- sum(is.na(ratings$Rating))
if (missing_ratings > 0) {
  cat("⚠️  Warning: Found", missing_ratings, "missing ratings\n")
}

# =============================================================================
# Core Analysis Functions
# =============================================================================

# Function to analyze ratings and create statement similarity matrix
analyze_ratings_and_similarity <- function(ratings, statements) {
  # Analyze ratings data and create similarity matrix for MDS analysis.
  #
  # This function:
  # 1. Calculates summary statistics for each statement and rating type
  # 2. Creates importance vs feasibility comparison
  # 3. Generates similarity matrix based on rating patterns
  # 4. Prepares data for multidimensional scaling
  #
  # Args:
  #   ratings: Dataframe containing participant ratings
  #   statements: Dataframe containing statement information
  #
  # Returns:
  #   List containing rating summary, importance-feasibility data, and similarity matrix
  cat("Analyzing ratings and creating similarity matrix...\n")
  
  # Calculate summary statistics by statement and rating type
  rating_summary <- ratings %>%
    group_by(StatementID, RatingType) %>%
    summarise(
      mean_rating = mean(Rating, na.rm = TRUE),
      median_rating = median(Rating, na.rm = TRUE),
      sd_rating = sd(Rating, na.rm = TRUE),
      n_ratings = n(),
      .groups = 'drop'
    ) %>%
    left_join(statements, by = "StatementID")
  
  # Create wide format for importance and feasibility comparison
  importance_feasibility <- ratings %>%
    pivot_wider(
      names_from = RatingType,
      values_from = Rating,
      values_fn = mean
    ) %>%
    group_by(StatementID) %>%
    summarise(
      Importance = mean(Importance, na.rm = TRUE),
      Feasibility = mean(Feasibility, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    left_join(statements, by = "StatementID")
  
  # Create similarity matrix based on rating patterns
  # Transform data so each row is a statement, columns are participant-ratingtype combinations
  rating_matrix <- ratings %>%
    mutate(ParticipantRatingType = paste0("P", ParticipantID, "_", RatingType)) %>%
    as.data.frame() %>%
    dplyr::select(StatementID, ParticipantRatingType, Rating) %>%
    arrange(StatementID) %>%
    pivot_wider(names_from = ParticipantRatingType, values_from = Rating) %>%
    column_to_rownames("StatementID") %>%
    as.matrix()

  # Calculate correlation matrix between statements
  # Higher correlation = more similar rating patterns
  similarity_matrix <- cor(t(rating_matrix), use = "pairwise.complete.obs")
  
  # Convert to distance matrix for MDS
  # Distance = 1 - |correlation| (absolute correlation to handle negative correlations)
  distance_matrix <- 1 - abs(similarity_matrix)
  diag(distance_matrix) <- 0  # Set diagonal to 0 (self-similarity)
  
  # Handle NA values by replacing with 0
  distance_matrix[is.na(distance_matrix)] <- 0
  
  cat("✅ Similarity matrix created:", nrow(similarity_matrix), "statements\n")
  
  return(list(
    rating_summary = rating_summary,
    importance_feasibility = importance_feasibility,
    similarity_matrix = similarity_matrix,
    distance_matrix = distance_matrix
  ))
}

# Function to perform multidimensional scaling
perform_mds_analysis <- function(distance_matrix, n_components = 2) {
  # Perform Multidimensional Scaling (MDS) on the distance matrix.
  #
  # MDS converts the high-dimensional rating patterns into 2D coordinates
  # that preserve the relative distances between statements. Statements with
  # similar rating patterns will be positioned closer together.
  #
  # Args:
  #   distance_matrix: Matrix of distances between statements
  #   n_components: Number of dimensions for MDS (default: 2)
  #
  # Returns:
  #   MDS coordinates and stress value
  cat("Performing Multidimensional Scaling (MDS)...\n")
  
  # Perform classical MDS using cmdscale
  mds_result <- cmdscale(distance_matrix, k = n_components, eig = TRUE)
  
  # Extract coordinates and stress
  mds_coords <- mds_result$points
  stress <- mds_result$GOF[1]  # Goodness of fit
  
  cat("✅ MDS completed with stress:", round(stress, 3), "\n")
  
  return(list(
    coordinates = mds_coords,
    stress = stress,
    eig = mds_result$eig
  ))
}

# Function to find optimal number of clusters
find_optimal_clusters <- function(mds_coords, max_k = 10) {
  # Find the optimal number of clusters using elbow method and silhouette analysis.
  #
  # This function evaluates different numbers of clusters (k) and selects the
  # optimal k based on:
  # 1. Elbow method: Where the within-cluster sum of squares (WSS) starts to level off
  # 2. Silhouette analysis: Average silhouette score for each k
  #
  # Args:
  #   mds_coords: 2D coordinates from MDS
  #   max_k: Maximum number of clusters to evaluate
  #
  # Returns:
  #   List containing optimal k, WSS scores, and silhouette scores
  cat("Finding optimal number of clusters...\n")
  
  # Standardize coordinates for clustering
  mds_scaled <- scale(mds_coords)
  
  # Calculate WSS for different k values
  wss <- numeric(max_k - 1)
  silhouette_scores <- numeric(max_k - 1)
  k_range <- 2:max_k
  
  for (i in seq_along(k_range)) {
    k <- k_range[i]
  
  # Perform k-means clustering
    kmeans_result <- kmeans(mds_scaled, centers = k, nstart = 25)
    
    # Calculate WSS
    wss[i] <- kmeans_result$tot.withinss
    
    # Calculate silhouette score (only if k > 1)
    if (k > 1) {
      silhouette_result <- silhouette(kmeans_result$cluster, dist(mds_scaled))
      silhouette_scores[i] <- mean(silhouette_result[, 3])
    }
  }
  
  # Find optimal k using elbow method
  optimal_k <- find_elbow_point(k_range, wss)
  
  cat("✅ Optimal number of clusters:", optimal_k, "\n")
  cat("   - Best silhouette score:", round(max(silhouette_scores), 3), "\n")
  
  return(list(
    optimal_k = optimal_k,
    wss = wss,
    silhouette_scores = silhouette_scores,
    k_range = k_range
  ))
}

# Helper function to find elbow point
find_elbow_point <- function(k_range, wss) {
  # Find the elbow point in the WSS curve using the second derivative method.
  #
  # The elbow point is where the rate of decrease in WSS starts to level off,
  # indicating diminishing returns from adding more clusters.
  #
  # Args:
  #   k_range: Range of k values evaluated
  #   wss: Within-cluster sum of squares for each k
  #
  # Returns:
  #   Optimal number of clusters (k value at elbow point)
  if (length(wss) < 3) {
    return(k_range[1])  # Default to first k if not enough points
  }
  
  # Calculate second derivative (rate of change of rate of change)
  first_diff <- diff(wss)
  second_diff <- diff(first_diff)
  
  # Find the point with maximum second derivative (sharpest bend)
  elbow_idx <- which.max(second_diff) + 2  # +2 because of double differencing
  
  # Ensure elbow_idx is within bounds
  elbow_idx <- min(elbow_idx, length(k_range))
  
  return(k_range[elbow_idx])
}

# Function to perform final clustering
perform_clustering <- function(mds_coords, n_clusters) {
  # Perform K-means clustering on the MDS coordinates.
  #
  # Args:
  #   mds_coords: 2D coordinates from MDS
  #   n_clusters: Number of clusters to create
  #
  # Returns:
  #   List containing cluster assignments and k-means model
  cat("Performing K-means clustering with", n_clusters, "clusters...\n")
  
  # Standardize coordinates
  mds_scaled <- scale(mds_coords)
  
  # Perform clustering
  kmeans_result <- kmeans(mds_scaled, centers = n_clusters, nstart = 25)
  
  cat("✅ Clustering completed:", length(unique(kmeans_result$cluster)), "clusters created\n")
  
  return(list(
    cluster_labels = kmeans_result$cluster,
    kmeans_model = kmeans_result
  ))
}

# =============================================================================
# Visualization Functions
# =============================================================================

# Function to create concept map visualization
create_concept_map <- function(mds_coords, cluster_labels, statements) {
  # Create the main concept map visualization.
  #
  # This is the primary visualization showing how statements are positioned
  # in 2D space based on their rating patterns, with color-coding by cluster.
  #
  # Args:
  #   mds_coords: 2D coordinates from MDS
  #   cluster_labels: Cluster assignments for each statement
  #   statements: Statement information
  cat("Creating concept map visualization...\n")
  
  # Prepare data for plotting
  plot_data <- data.frame(
    x = mds_coords[, 1],
    y = mds_coords[, 2],
    cluster = factor(cluster_labels),
    statement_id = statements$StatementID,
    statement_text = statements$StatementText
  )
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = x, y = y, color = cluster)) +
    geom_point(size = 4, alpha = 0.8) +
    geom_text_repel(
      aes(label = statement_id),
      size = 3,
      fontface = "bold",
      max.overlaps = 20
    ) +
    scale_color_viridis_d(name = "Cluster") +
    labs(
      title = "Concept Map: Multidimensional Scaling with Clusters",
      subtitle = "Statements positioned by similarity in rating patterns",
      x = "MDS Dimension 1",
      y = "MDS Dimension 2"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom"
    ) +
    guides(color = guide_legend(nrow = 1))
  
  # Save the plot
  ggsave(
    file.path(output_dir, "concept_map.png"),
    plot = p,
    width = 12,
    height = 10,
    dpi = 300
  )
  
  cat("✅ Concept map saved\n")
  return(p)
}

# Function to create importance vs feasibility plot
create_importance_feasibility_plot <- function(importance_feasibility) {
  # Create importance vs feasibility scatter plot.
  #
  # This visualization shows the relationship between importance and feasibility
  # ratings, with statements positioned based on their mean ratings on both dimensions.
  #
  # Args:
  #   importance_feasibility: Dataframe with importance and feasibility ratings
  cat("Creating importance vs feasibility plot...\n")
  
  # Calculate mean lines
  mean_importance <- mean(importance_feasibility$Importance, na.rm = TRUE)
  mean_feasibility <- mean(importance_feasibility$Feasibility, na.rm = TRUE)
  
  # Create the plot
  p <- ggplot(importance_feasibility, aes(x = Importance, y = Feasibility)) +
    geom_point(size = 3, alpha = 0.7, color = "steelblue") +
    geom_text_repel(
      aes(label = StatementID),
      size = 3,
      fontface = "bold",
      max.overlaps = 15
    ) +
    # Add mean lines
    geom_vline(xintercept = mean_importance, color = "red", linetype = "dashed", alpha = 0.7) +
    geom_hline(yintercept = mean_feasibility, color = "red", linetype = "dashed", alpha = 0.7) +
    # Add quadrant labels
    annotate("text", x = 0.1, y = 0.9, label = "High Importance\nLow Feasibility", 
             hjust = 0, vjust = 1, size = 4, color = "darkblue",
             fontface = "bold") +
    annotate("text", x = 0.9, y = 0.9, label = "High Importance\nHigh Feasibility", 
             hjust = 1, vjust = 1, size = 4, color = "darkgreen",
             fontface = "bold") +
    annotate("text", x = 0.1, y = 0.1, label = "Low Importance\nLow Feasibility", 
             hjust = 0, vjust = 0, size = 4, color = "darkgray",
             fontface = "bold") +
    annotate("text", x = 0.9, y = 0.1, label = "Low Importance\nHigh Feasibility", 
             hjust = 1, vjust = 0, size = 4, color = "darkorange",
             fontface = "bold") +
    labs(
      title = "Importance vs Feasibility: Strategic Quadrants",
      subtitle = paste("Mean Importance:", round(mean_importance, 2), 
                      "| Mean Feasibility:", round(mean_feasibility, 2)),
      x = "Importance Rating",
      y = "Feasibility Rating"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(size = 12, face = "bold")
    ) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1))
  
  # Save the plot
  ggsave(
    file.path(output_dir, "importance_vs_feasibility.png"),
    plot = p,
    width = 12,
    height = 10,
    dpi = 300
  )
  
  cat("✅ Importance vs feasibility plot saved\n")
  return(p)
}

# Function to create rating distribution plots
create_rating_distribution <- function(ratings) {
  # Create rating distribution histograms for importance and feasibility.
  #
  # This visualization shows the distribution of ratings across all statements
  # and participants, helping to understand the overall rating patterns.
  #
  # Args:
  #   ratings: Dataframe containing all ratings
  cat("Creating rating distribution plots...\n")
  
  # Create importance rating plot
  p1 <- ratings %>%
    filter(RatingType == "Importance") %>%
    ggplot(aes(x = Rating)) +
    geom_histogram(bins = 6, fill = "steelblue", alpha = 0.7, color = "black") +
    geom_vline(xintercept = mean(ratings$Rating[ratings$RatingType == "Importance"], na.rm = TRUE),
               color = "red", linetype = "dashed", size = 1) +
    labs(
      title = "Distribution of Importance Ratings",
      x = "Importance Rating",
      y = "Frequency"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold")
    )
  
  # Create feasibility rating plot
  p2 <- ratings %>%
    filter(RatingType == "Feasibility") %>%
    ggplot(aes(x = Rating)) +
    geom_histogram(bins = 6, fill = "orange", alpha = 0.7, color = "black") +
    geom_vline(xintercept = mean(ratings$Rating[ratings$RatingType == "Feasibility"], na.rm = TRUE),
               color = "red", linetype = "dashed", size = 1) +
    labs(
      title = "Distribution of Feasibility Ratings",
      x = "Feasibility Rating",
      y = "Frequency"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold")
    )
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, ncol = 2, 
                               top = textGrob("Rating Distributions", 
                                             gp = gpar(fontsize = 16, fontface = "bold")))
  
  # Save the plot
  ggsave(
    file.path(output_dir, "rating_distribution.png"),
    plot = combined_plot,
    width = 15,
    height = 6,
    dpi = 300
  )
  
  cat("✅ Rating distribution plots saved\n")
  return(combined_plot)
}

# Function to create cluster analysis plots
create_cluster_analysis_plots <- function(mds_coords) {
  # Create cluster analysis plots showing WSS and silhouette analysis.
  #
  # These plots help visualize the process of finding the optimal number
  # of clusters and validate the clustering solution.
  #
  # Args:
  #   mds_coords: 2D coordinates from MDS
  cat("Creating cluster analysis plots...\n")
  
  # Get cluster analysis data
  cluster_analysis <- find_optimal_clusters(mds_coords)
  
  # Create WSS plot
  p1 <- data.frame(
    k = cluster_analysis$k_range,
    wss = cluster_analysis$wss
  ) %>%
    ggplot(aes(x = k, y = wss)) +
    geom_line(size = 1, color = "blue") +
    geom_point(size = 3, color = "blue") +
    geom_vline(xintercept = cluster_analysis$optimal_k, 
               color = "red", linetype = "dashed", size = 1) +
    labs(
      title = "Elbow Method for Optimal k Selection",
      x = "Number of Clusters (k)",
      y = "Within-Cluster Sum of Squares (WSS)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold")
    )
  
  # Create silhouette plot
  p2 <- data.frame(
    k = cluster_analysis$k_range,
    silhouette = cluster_analysis$silhouette_scores
  ) %>%
    ggplot(aes(x = k, y = silhouette)) +
    geom_line(size = 1, color = "green") +
    geom_point(size = 3, color = "green") +
    geom_vline(xintercept = cluster_analysis$optimal_k, 
               color = "red", linetype = "dashed", size = 1) +
    labs(
      title = "Silhouette Analysis for Optimal k Selection",
      x = "Number of Clusters (k)",
      y = "Average Silhouette Score"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12, face = "bold")
    )
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, ncol = 2,
                               top = textGrob("Cluster Analysis: Finding Optimal Number of Clusters",
                                             gp = gpar(fontsize = 16, fontface = "bold")))
  
  # Save the plot
  ggsave(
    file.path(output_dir, "cluster_analysis.png"),
    plot = combined_plot,
    width = 15,
    height = 6,
    dpi = 300
  )
  
  cat("✅ Cluster analysis plots saved\n")
  return(combined_plot)
}

# Function to create similarity heatmap
create_similarity_heatmap <- function(similarity_matrix) {
  # Create similarity matrix heatmap.
  #
  # This visualization shows the correlation matrix between statements,
  # helping to identify groups of statements with similar rating patterns.
  #
  # Args:
  #   similarity_matrix: Correlation matrix between statements
  cat("Creating similarity heatmap...\n")
  
  # Create heatmap using corrplot
  png(file.path(output_dir, "similarity_heatmap.png"), 
      width = 12, height = 10, units = "in", res = 300)
  
  corrplot(
    similarity_matrix,
    method = "color",
    type = "upper",
    order = "hclust",
    tl.pos = "n",  # No text labels for cleaner look
    col = colorRampPalette(c("blue", "white", "red"))(100),
    title = "Statement Similarity Matrix",
    mar = c(0, 0, 2, 0)
  )
  
  dev.off()
  
  cat("✅ Similarity heatmap saved\n")
}

# =============================================================================
# Statistical Analysis Functions
# =============================================================================

# Function to generate summary statistics
generate_summary_statistics <- function(ratings, statements, cluster_labels, importance_feasibility) {
  # Generate comprehensive summary statistics for the concept mapping analysis.
  #
  # This function creates summary tables and statistics including:
  # - Statement-level statistics (mean ratings, cluster assignments)
  # - Overall correlation between importance and feasibility
  # - Data quality metrics
  #
  # Args:
  #   ratings: All rating data
  #   statements: Statement information
  #   cluster_labels: Cluster assignments
  #   importance_feasibility: Importance vs feasibility data
  #
  # Returns:
  #   List containing summary statistics
  cat("Generating summary statistics...\n")
  
  # Calculate overall correlation
  correlation <- cor(importance_feasibility$Importance, 
                    importance_feasibility$Feasibility, 
                    use = "complete.obs")
  
  # Create statement summary with cluster information
  statement_summary <- statements %>%
    mutate(Cluster = cluster_labels) %>%
    left_join(
      importance_feasibility %>% dplyr::select(StatementID, Importance, Feasibility),
      by = "StatementID"
    )
  
  # Calculate cluster statistics
  cluster_stats <- statement_summary %>%
    group_by(Cluster) %>%
    summarise(
      n_statements = n(),
      mean_importance = mean(Importance, na.rm = TRUE),
      mean_feasibility = mean(Feasibility, na.rm = TRUE),
      .groups = 'drop'
    )
  
  # Create overall summary
  summary_stats <- list(
    total_statements = nrow(statements),
    total_participants = length(unique(ratings$ParticipantID)),
    total_ratings = nrow(ratings),
    correlation_importance_feasibility = correlation,
    n_clusters = length(unique(cluster_labels)),
    cluster_statistics = cluster_stats,
    statement_summary = statement_summary
  )
  
  # Save summary statistics
  write_csv(statement_summary, file.path(output_dir, "summary_statistics.csv"))
  
  cat("✅ Summary statistics generated:\n")
  cat("   - Correlation (Importance vs Feasibility):", round(correlation, 3), "\n")
  cat("   - Total statements:", summary_stats$total_statements, "\n")
  cat("   - Total participants:", summary_stats$total_participants, "\n")
  cat("   - Total ratings:", summary_stats$total_ratings, "\n")
  cat("   - Number of clusters:", summary_stats$n_clusters, "\n")
  
  return(summary_stats)
}

# =============================================================================
# Main Analysis Workflow
# =============================================================================

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("CONCEPT MAPPING ANALYSIS - R IMPLEMENTATION\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Step 1: Analyze ratings and create similarity matrix
analysis_results <- analyze_ratings_and_similarity(ratings, statements)

# Step 2: Perform MDS analysis
mds_results <- perform_mds_analysis(analysis_results$distance_matrix)

# Step 3: Find optimal clusters
cluster_analysis <- find_optimal_clusters(mds_results$coordinates)

# Step 4: Perform final clustering
clustering_results <- perform_clustering(mds_results$coordinates, cluster_analysis$optimal_k)

# Step 5: Create visualizations
cat("\nCreating visualizations...\n")

# Concept map
concept_map <- create_concept_map(
  mds_results$coordinates, 
  clustering_results$cluster_labels, 
  statements
)

# Importance vs feasibility plot
importance_feasibility_plot <- create_importance_feasibility_plot(
  analysis_results$importance_feasibility
)

# Rating distribution plots
rating_distribution_plots <- create_rating_distribution(ratings)

# Cluster analysis plots
cluster_analysis_plots <- create_cluster_analysis_plots(mds_results$coordinates)

# Similarity heatmap
create_similarity_heatmap(analysis_results$similarity_matrix)

# Step 6: Generate summary statistics
summary_stats <- generate_summary_statistics(
  ratings, 
  statements, 
  clustering_results$cluster_labels,
  analysis_results$importance_feasibility
)

# Step 7: Create final results dataframe
results_df <- statements %>%
  mutate(
    MDS_Dim1 = mds_results$coordinates[, 1],
    MDS_Dim2 = mds_results$coordinates[, 2],
    Cluster = clustering_results$cluster_labels
  ) %>%
  left_join(
    analysis_results$importance_feasibility %>% select(StatementID, Importance, Feasibility),
    by = "StatementID"
  )

# Save final results
write_csv(results_df, file.path(output_dir, "statements_with_clusters.csv"))

# =============================================================================
# Final Summary
# =============================================================================

cat("\n" + paste(rep("=", 60), collapse = ""), "\n")
cat("ANALYSIS COMPLETED SUCCESSFULLY!\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("📊 Results saved to:", output_dir, "\n")
cat("📈 Visualizations created:", length(list.files(output_dir, pattern = "\\.png$")), "files\n")
cat("📋 Summary:", summary_stats$total_statements, "statements in", summary_stats$n_clusters, "clusters\n")
cat("🔗 Correlation (Importance vs Feasibility):", round(summary_stats$correlation_importance_feasibility, 3), "\n")
cat("📁 Check the output directory for all results and visualizations!\n")

# Return complete results for further analysis
invisible(list(
  statements = statements,
  ratings = ratings,
  demographics = demographics,
  mds_coords = mds_results$coordinates,
  cluster_labels = clustering_results$cluster_labels,
  similarity_matrix = analysis_results$similarity_matrix,
  optimal_k = cluster_analysis$optimal_k,
  kmeans_model = clustering_results$kmeans_model,
  summary_stats = summary_stats,
  results_df = results_df
)) 