# How to Run RCMap with Your Workshop Data

## Your Data Files
All your RCMap input files are ready in: `data/rcmap_workshop/`

- **Statements.csv** - All 100 statements from your workshop
- **SortedCards_example.csv** - Example grouping data (from Survey Preview participant)
- **Ratings_example.csv** - Example rating data (from Survey Preview participant)  
- **Demographics.csv** - Participant information

## Method 1: Interactive R Session (Recommended)

1. **Open R in your terminal:**
   ```bash
   R
   ```

2. **Load RCMap and open the menu:**
   ```r
   library(RCMap)
   RCMapMenu()
   ```

3. **In the RCMap menu:**
   - Choose "Choose the data folder"
   - Navigate to: `data/rcmap_workshop/`
   - Select which analyses to run:
     - Statement Report
     - Sorter Report (grouping analysis)
     - Rater Report (rating analysis)
     - Rating Summary
     - Concept Maps
     - Cluster Analysis

## Method 2: RStudio (if you have it)

1. **Open RStudio**
2. **Load RCMap:**
   ```r
   library(RCMap)
   RCMapMenu()
   ```
3. **Follow the same steps as above**

## What RCMap Will Generate

- **Concept maps** showing relationships between statements
- **Cluster analysis** of grouped statements  
- **Rating summaries** and visualizations
- **Participant reports** showing individual patterns
- **PDF reports** with all analysis results

## Important Notes

- **Current data**: You're working with example data from the Survey Preview participant
- **For full analysis**: You'd need more participants to complete both grouping and rating tasks
- **The 100 statements**: Are all properly numbered and ready for analysis

## Troubleshooting

If you get errors about missing packages, run:
```r
install.packages(c("smacof", "plotrix", "colorspace", "e1071", "factoextra", "ggplot2", "crayon", "ape", "tcltk"))
```

Then try loading RCMap again:
```r
library(RCMap)
RCMapMenu()
``` 