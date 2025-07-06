#!/usr/bin/env Rscript

# Transform Q2 Qualtrics Data to RCMap Format
# Uses provided statement text

library(dplyr)
library(readr)
library(stringr)
library(tidyr)

# Configuration
input_file <- "data/cleaned_q2_q3_only.csv"
output_dir <- "data/rcmap_q2"

# Provided statement text (copy-paste from user, one per line)
statements <- c(
  "Human oversight during implementation in early days",
  "Concerns with confidentiality",
  "Concerns with patient acceptability",
  "Efficiency improved for documentation",
  "False confidence",
  "AI could provide alternate perspectives",
  "AI could perpetuate existing biases",
  "Minimizing bias and increasing performance and accuracy",
  "Ensuring accountability and oversight of AI",
  "Concerns with digital literacy and digital divide barriers",
  "Better detection (of moles, cancer lumps)",
  "Diversification of AI training data and sources",
  "Too much bureaucracy slowing progress",
  "Concerns with liability",
  "AI could decrease false positives",
  "Proactive preventative care",
  "Benefits in equitable access",
  "AI could free up time for pt care",
  "Developing a rigorous privacy and quality assurance framework",
  "AI could be misused",
  "Helps with generation of ideas can act as a sounding board",
  "Benefits with accessibility & communication (translating languages lay terms)",
  "AI allows access vast information very quickly to help diagnosis",
  "Benefits in early distress screening",
  "Establishing proper consent",
  "Concerns with job security",
  "Benefits with patient access to accurate information",
  "Concerns about personalizing care",
  "Data governance policies and regulations to enhance data quality and accountability",
  "Savings in time, can integrate different information and improve in treatment especially when there is less number of human specialist",
  "AI could remove human bias",
  "Regulation framework and a good understanding of the medico-legal implications",
  "AI could lead to excessive human delegation, assuming that AI will take care of it!",
  "Knowledge repositing for learning and research",
  "AI could be of concern regarding its ongoing validation",
  "AI could be of concern to patient privacy",
  "Consultation recording and transcription",
  "Ensuring cybersecurity of AI tools",
  "Educating end-users about properly navigating AI",
  "Concerns with protection of personal health identifiers",
  "Facilitation of clinical research",
  "Improve access to services",
  "Ensuring accessibility and translation",
  "AI could help improve accuracy",
  "Embed research / quality improvement into all areas",
  "'Garbage in, garbage out' – data that the AI model is trained on must be good, clean, large and diverse",
  "Concerns with attrition of skills in healthcare providers",
  "Preservation of human touch",
  "Loss of human connection/interactions",
  "Summarization of large amounts of text or data – increases digestability/highlighting key points",
  "Collate large databases",
  "AI could engender issues with trust",
  "Pooling large volume of data can improve accuracy of AI system and allow for more timely diagnosis and treatment recommendation",
  "Concerns with responsibility accountability of AI recommendations",
  "AI hallucinations",
  "Multidisciplinary collaboration and training in developing AI models",
  "24/7 availability, AI doesn't get tired or brain fog",
  "Noticing patterns and new things humans haven't noticed",
  "Prediction/prognosis/treatment recommendations",
  "Concerns with standards for quality, accuracy, etc.",
  "Concerns with power outages and down time procedures",
  "Increased opportunities for innovation",
  "Distress is nuanced and often detected in the unsaid of human communication and in a trusting therapeutic relationship",
  "Personalization of AI support tools for patients",
  "Concerns with difficulty to know older data for training and missing new breakthrough information",
  "Concerns with consent of patient",
  "Racism bias through research (history of medical research)",
  "Phased roll-out: Start with simple, 'low stakes' tasks continue in stages",
  "Reduce admin burden",
  "Potential in navigation and directing to self-management resources",
  "Over reliance by physicians and reduced problem solving skills",
  "Ability to make text/conversation more digestable for patients (less technical)",
  "Ensuring clinical validation of AI tools",
  "Concerns with maintaining up to date information, sources and programming",
  "Data privacy concerns",
  "Triage diagnostic results for clinicians",
  "Concerns with going too far and can't come back",
  "Control of data by corporations",
  "AI could help address the human health resources issue",
  "Faster assessments to inform decision-making",
  "Concerns with transparency",
  "AI could make mistakes",
  "Concerns with stability of the system (during virus attack, power outage, system interruption)",
  "Live translation and/or transcription of conversations",
  "Establishing guidelines for 'best practices' for training AI + clinical validation",
  "Lack of training/knowledge for those using the AI tool in the real world",
  "Can existing healthcare servers support the use of AI?",
  "Ensuring interpretability of AI tools",
  "Allow clinicians to focus on tasks requiring their expertise",
  "Identifying and focusing on high-priority areas, such as prevention",
  "Concerns that data is held by private companies",
  "Maintaining competency with changing best practices and approved standards",
  "Minority groups being marginalized",
  "Patient/parent/caregiver/provider acceptability",
  "Data ownership and standards and guidelines on AI developed are not established",
  "Possibility of data breaches / data leaks",
  "Potential for widening differential diagnosis, widening perspective of patient",
  "Lists possible outcome and risks for the patient",
  "Cybersecurity concerns",
  "Automation of routine tasks"
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Reading cleaned Q2 data...\n")
data <- read_csv(input_file, show_col_types = FALSE)

# Identify Q2 columns (pattern: Q2.)
col_names <- names(data)
q2_cols <- grep("^Q2\\.", col_names, value = TRUE)
meta_cols <- setdiff(col_names, q2_cols)

# --- Statements.csv ---
cat("Creating Statements.csv...\n")
statements_df <- data.frame(
  StatementID = seq_along(statements),
  StatementText = statements,
  stringsAsFactors = FALSE
)
write_csv(statements_df, file.path(output_dir, "Statements.csv"))

# --- Ratings_Q2.csv ---
cat("Creating Ratings_Q2.csv...\n")
ratings_q2 <- data %>%
  mutate(ParticipantID = row_number()) %>%
  select(ParticipantID, all_of(q2_cols)) %>%
  pivot_longer(
    cols = all_of(q2_cols),
    names_to = "QCol",
    values_to = "Rating"
  ) %>%
  mutate(
    StatementID = as.integer(str_extract(QCol, "\\d+$"))  # Use last number as statement index
  ) %>%
  select(ParticipantID, StatementID, Rating) %>%
  filter(!is.na(Rating) & Rating != "" & Rating != "NA")
write_csv(ratings_q2, file.path(output_dir, "Ratings_Q2.csv"))

# --- Demographics.csv ---
cat("Creating Demographics.csv...\n")
demographics <- data %>%
  mutate(ParticipantID = row_number()) %>%
  select(ParticipantID, all_of(meta_cols))
write_csv(demographics, file.path(output_dir, "Demographics.csv"))

cat("All RCMap input files for Q2 created in:", output_dir, "\n")
cat("You can now run RCMap as usual on Q2 ratings.\n") 